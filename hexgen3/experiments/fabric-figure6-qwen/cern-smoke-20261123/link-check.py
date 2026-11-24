"""Measure each directed CUDA tensor transfer on the three smoke-test ranks."""

from datetime import timedelta
import json
from time import perf_counter

import torch
import torch.distributed as dist

from hexgen2.nccl_runtime import runtime_version
from hexgen2.transport import recv_tensor, send_tensor

device = torch.device("cuda:0")
torch.cuda.set_device(device)
assert torch.cuda.is_bf16_supported()
dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=3))
rank = dist.get_rank()
size = dist.get_world_size()
payload = torch.empty(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device="cuda")
results = []
try:
    for source in range(size):
        for target in range(size):
            if source == target:
                continue
            payload.fill_(rank + 1)
            for _ in range(2):
                if rank == source:
                    send_tensor(payload, target)
                elif rank == target:
                    recv_tensor(payload, source)
            torch.cuda.synchronize()
            dist.barrier(device_ids=[0])
            started = perf_counter()
            iterations = 8
            for _ in range(iterations):
                if rank == source:
                    send_tensor(payload, target)
                elif rank == target:
                    recv_tensor(payload, source)
            torch.cuda.synchronize()
            elapsed = perf_counter() - started
            if rank == target:
                assert torch.all(payload == source + 1).item()
                results.append({
                    "source": source, "target": target, "bytes": payload.nbytes,
                    "iterations": iterations, "seconds": elapsed,
                    "gbits_per_second": payload.nbytes * iterations * 8e-9 / elapsed,
                })
            dist.barrier(device_ids=[0])
    print(json.dumps({"rank": rank, "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__, "cuda": torch.version.cuda,
                      "nccl": runtime_version(), "results": results}), flush=True)
finally:
    dist.destroy_process_group()
