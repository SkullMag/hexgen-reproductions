"""Check Qwen prefill-to-decode KV transfer on the CERN three-rank placement."""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Qwen2MoeConfig

from .comparison_model import State
from .qwen_stage import QwenStage
from .transport import recv_packet, send_packet


def qualify(model_dir, cases, device):
    rank = dist.get_rank()
    layers = Qwen2MoeConfig.from_pretrained(model_dir, local_files_only=True).num_hidden_layers
    split = layers // 2
    partitions = [(0, layers), (0, split), (split, layers)]
    stage = QwenStage(model_dir, *partitions[rank], device=device)
    results = []
    for case in cases:
        prompt = torch.tensor([case['input_ids']], dtype=torch.long, device=device)
        length, count = prompt.shape[1], case['max_new_tokens']
        if not 0 < length or not 0 < count or length + count > stage.hf.max_position_embeddings:
            raise ValueError('Smoke request is outside the checkpoint context limit')
        if rank == 0:
            reference = State(length + count, prompt.shape[0])
            first = stage(prompt, reference).argmax(-1)
            reference.sequence_len_offset = length
            for peer, (begin, end) in enumerate(partitions[1:], start=1):
                live = {str(i): reference.key_value_memory_dict[i][:, :length] for i in range(begin, end)}
                send_packet({'first': first, **live}, {'id': case['id']}, peer)
            token = first
            expected = [token]
            for _ in range(count - 1):
                token = stage(token[:, None], reference).argmax(-1)
                reference.sequence_len_offset += 1
                expected.append(token)
            actual, metadata = recv_packet(2)
            matches = metadata['id'] == case['id'] and torch.equal(torch.stack(expected, dim=1), actual['tokens'])
            results.append({'id': case['id'], 'matched': matches, 'output_tokens': count})
        else:
            tensors, metadata = recv_packet(0)
            if metadata['id'] != case['id']:
                raise ValueError('Received KV cache for a different request')
            token, state = stage.restore(tensors, length, count)
            generated = [token]
            for _ in range(count - 1):
                token, _ = stage.step(token[:, None], state, None, 1, 2)
                state.sequence_len_offset += 1
                generated.append(token)
            if rank == 2:
                send_packet({'tokens': torch.stack(generated, dim=1)}, {'id': case['id']}, 0)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', 0)) != 3:
        parser.error('Set WORLD_SIZE=3 with one process per rank')
    cases = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    if not cases:
        parser.error('Trace is empty')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    backend = 'gloo'
    if device.type == 'cuda':
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        torch.cuda.set_device(device)
        backend = 'nccl'
    dist.init_process_group(backend, timeout=timedelta(minutes=10))
    try:
        results = qualify(args.model_dir, cases, device)
        if dist.get_rank() == 0:
            report = {'device': device.type, 'backend': backend, 'requests': results,
                      'passed': all(row['matched'] for row in results)}
            with args.output.open('x') as f:
                json.dump(report, f, indent=2)
                f.write('\n')
            if not report['passed']:
                raise SystemExit(2)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
