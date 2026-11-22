"""Check staged Qwen prefill and transferred-cache decode against the full model."""

from pathlib import Path
import tempfile
import unittest

import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
from transformers.cache_utils import DynamicCache

from hexgen2.comparison_model import State
from hexgen2.qwen_stage import QwenStage


class QwenStageTest(unittest.TestCase):
    @torch.inference_mode()
    def test_sharded_bf16_prefill_and_decode(self):
        torch.manual_seed(917)
        config = Qwen2MoeConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            max_position_embeddings=32, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, num_experts=4,
            num_experts_per_tok=2, moe_intermediate_size=12,
            shared_expert_intermediate_size=16,
        )
        config._attn_implementation = 'sdpa'
        reference = Qwen2MoeForCausalLM(config).eval().to(torch.bfloat16)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            reference.save_pretrained(model_dir, max_shard_size='12KB')
            prefill = QwenStage(model_dir, 0, 4, device='cpu')
            head = QwenStage(model_dir, 0, 2, device='cpu')
            tail = QwenStage(model_dir, 2, 4, device='cpu')

        inputs = torch.tensor([[1, 8, 16, 3], [1, 5, 7, 11]])
        prefix_len = inputs.shape[1]
        next_tokens = 3
        prefill_state = State(prefix_len, inputs.shape[0])
        logits = prefill(inputs, prefill_state)
        expected = reference(inputs, use_cache=True, past_key_values=DynamicCache())
        torch.testing.assert_close(logits, expected.logits[:, -1])
        first_token = logits.argmax(-1)
        decode_states = {}
        for stage in (head, tail):
            tensors = {
                str(i): prefill_state.key_value_memory_dict[i]
                for i in range(stage.start, stage.stop)
            }
            token, state = stage.restore(
                {'first': first_token, **tensors}, prefix_len, next_tokens,
            )
            self.assertTrue(torch.equal(token, first_token))
            decode_states[stage] = state

        for _ in range(next_tokens - 1):
            expected = reference(token[:, None], past_key_values=expected.past_key_values)
            hidden = head(token[:, None], decode_states[head])
            logits = tail(hidden, decode_states[tail])
            # SDPA may round differently for packed versus strided BF16 KV buffers.
            torch.testing.assert_close(logits, expected.logits[:, -1], rtol=0.02, atol=0.001)
            token = logits.argmax(-1)
            self.assertTrue(torch.equal(token, expected.logits[:, -1].argmax(-1)))
            for state in decode_states.values():
                state.sequence_len_offset += 1
