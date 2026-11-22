"""Qwen1.5-MoE model stages for the HexGen-2 reproduction adapter."""

from collections import defaultdict
import json
from pathlib import Path

from safetensors import safe_open
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen2MoeConfig
from transformers.cache_utils import Cache
from transformers.models.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeDecoderLayer, Qwen2MoeRMSNorm, Qwen2MoeRotaryEmbedding,
)

from .comparison_model import NativeStage


class StageCache(Cache):
    """Expose the live prefix of a stage's native (B, S, 2, H, D) KV buffers."""

    def __init__(self, state):
        super().__init__()
        self.state = state

    def update(self, keys, values, layer_idx, cache_kwargs=None):
        start = self.state.sequence_len_offset
        stop = start + keys.shape[2]
        if stop > self.state.max_sequence_len:
            raise ValueError('Sequence exceeds allocated KV cache')
        buffer = self.state.key_value_memory_dict.get(layer_idx)
        if buffer is None:
            buffer = keys.new_empty((keys.shape[0], self.state.max_sequence_len,
                                     2, keys.shape[1], keys.shape[3]))
            self.state.key_value_memory_dict[layer_idx] = buffer
        buffer[:, start:stop, 0].copy_(keys.transpose(1, 2))
        buffer[:, start:stop, 1].copy_(values.transpose(1, 2))
        return buffer[:, :stop, 0].transpose(1, 2), buffer[:, :stop, 1].transpose(1, 2)

    def get_seq_length(self, layer_idx=0):
        return self.state.sequence_len_offset


class QwenStage(NativeStage):
    """Load only the owned layers, using Hugging Face Qwen2-MoE compute blocks."""

    def __init__(self, model_dir, start, stop, max_batch=1, *, device='cuda',
                 dtype=torch.bfloat16):
        nn.Module.__init__(self)
        self.hf = Qwen2MoeConfig.from_pretrained(model_dir, local_files_only=True)
        if self.hf.model_type != 'qwen2_moe' or self.hf.use_sliding_window:
            raise ValueError('Qwen stages require Qwen2-MoE with full causal attention')
        if not 0 <= start < stop <= self.hf.num_hidden_layers:
            raise ValueError('Invalid Qwen layer partition')
        self.hf._attn_implementation = 'sdpa'
        self.start, self.stop, self.dtype = start, stop, dtype
        self.device = torch.device(device)
        self.audit_directory = self.audit_context = None
        self.audit_sequence = 0
        self.embed_tokens = self.norm = self.lm_head = None
        with torch.device('meta'):
            if start == 0:
                self.embed_tokens = nn.Embedding(self.hf.vocab_size, self.hf.hidden_size,
                                                 padding_idx=self.hf.pad_token_id)
            self.layers = nn.ModuleDict({
                str(i): Qwen2MoeDecoderLayer(self.hf, i) for i in range(start, stop)
            })
            if stop == self.hf.num_hidden_layers:
                self.norm = Qwen2MoeRMSNorm(self.hf.hidden_size, eps=self.hf.rms_norm_eps)
                self.lm_head = nn.Linear(self.hf.hidden_size, self.hf.vocab_size, bias=False)
        self._load(Path(model_dir))
        self.rotary_emb = Qwen2MoeRotaryEmbedding(config=self.hf, device=self.device)
        self.output_buffer = torch.empty(max_batch * self.hf.max_position_embeddings,
                                         dtype=torch.int64, pin_memory=self.device.type == 'cuda')
        self.source = {
            'system': 'hexgen2',
            'model_adapter': 'Transformers Qwen2-MoE decoder with Torch SDPA',
            'layers': list(range(start, stop)), 'parameter_dtype': str(dtype),
            'stage_parameter_bytes': sum(p.numel() * p.element_size() for p in self.parameters()),
        }
        self.eval()

    def _load(self, directory):
        index = directory / 'model.safetensors.index.json'
        weight_map = json.loads(index.read_text())['weight_map'] if index.exists() else {}
        files = defaultdict(list)
        for name in self.state_dict():
            checkpoint_key = name if name == 'lm_head.weight' else f'model.{name}'
            if name == 'lm_head.weight' and self.hf.tie_word_embeddings:
                checkpoint_key = 'model.embed_tokens.weight'
            shard = weight_map[checkpoint_key] if weight_map else 'model.safetensors'
            files[shard].append((name, checkpoint_key))
        weights = {}
        for shard, keys in files.items():
            with safe_open(directory / shard, framework='pt', device='cpu') as f:
                for name, key in keys:
                    weights[name] = f.get_tensor(key).to(device=self.device, dtype=self.dtype)
        self.load_state_dict(weights, strict=True, assign=True)

    @torch.inference_mode()
    def forward(self, inputs, state):
        hidden = self.embed_tokens(inputs) if self.embed_tokens is not None else inputs
        offset = state.sequence_len_offset
        length = hidden.shape[1]
        if offset and length != 1:
            raise ValueError('Qwen adapter expects full prefill or single-token decode')
        positions = torch.arange(offset, offset + length, device=hidden.device)
        position_ids = positions.unsqueeze(0)
        rotary = self.rotary_emb(hidden, position_ids)
        cache = StageCache(state)
        for layer in self.layers.values():
            hidden = layer(hidden, position_ids=position_ids, past_key_value=cache,
                           use_cache=True, cache_position=positions, position_embeddings=rotary)[0]
        if self.lm_head is None:
            return hidden
        return self.lm_head(self.norm(hidden[:, -1]))

    @torch.inference_mode()
    def step(self, ids, state, pair, head_rank, tail_rank):
        batch, length = ids.shape
        if self.embed_tokens is not None:
            hidden = self(ids, state)
            dist.send(hidden.contiguous(), dst=tail_rank, group=pair)
            token = torch.empty(batch, dtype=torch.long, device=self.device)
            dist.recv(token, src=tail_rank, group=pair)
            return token, None

        hidden = torch.empty((batch, length, self.hf.hidden_size),
                             dtype=self.dtype, device=self.device)
        dist.recv(hidden, src=head_rank, group=pair)
        logits = self(hidden, state)
        token = logits.argmax(-1)
        if self.audit_directory is not None:
            torch.save({'context': self.audit_context, 'sequence_offset': state.sequence_len_offset,
                        'logits': logits.float().cpu()},
                       self.audit_directory / f'audit-{self.audit_sequence:06}.pt')
            self.audit_sequence += 1
        dist.send(token.contiguous(), dst=head_rank, group=pair)
        return token, logits
