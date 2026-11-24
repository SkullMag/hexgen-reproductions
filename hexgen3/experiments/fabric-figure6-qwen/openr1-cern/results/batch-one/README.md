# OpenR1-Math decode batch size one control

The control replayed the three frozen 0.85× schedules at 0.01242418 requests/s. Each arm used a 480-second window with a 450-second HTTP timeout and a 460-second drain allowance. HexGen-2 went first for seeds 61 and 997, and HexGen-3 went first for seed 305. The analyzer verified the frozen prompt, output budget, scheduled offset, trace hash, arrival hash, and order for every pair.

FABRIC slice `hexgen-batch-one-20261124-014644` used the same physical placement as the original benchmark. The A40 prefill host was `cern-w1`. The A30 hosts were `cern-w3` and `cern-w6`. The separate CPU client ran on `cern-w5`. Both systems used BF16 Qwen1.5-MoE-A2.7B-Chat revision `ec052fda178e241c7c443468d2fa1db6618996be` with checkpoint manifest `5d322584d22c982c73a014afebfbc41072351fa35c837ecae667ae95310931f1`. The GPU driver was 580.178.04. The runtime versions matched the [original protocol](../../README.md#fixed-inputs).

| Settings | Runtime | Finished in window | Finished after drain | Requests/min in window | P99 (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Original settings | HexGen-2 | 18 / 27 | 27 / 27 | 0.750 | 339.78 |
| Original settings | HexGen-3 | 27 / 27 | 27 / 27 | 1.125 | 34.82 |
| Both decode caps set to one | HexGen-2 | 19 / 27 | 27 / 27 | 0.792 | 325.88 |
| Both decode caps set to one | HexGen-3 | 26 / 27 | 27 / 27 | 1.083 | 67.86 |

P99 is the maximum of the 27 successful requests in each arm. With the one-request cap, HexGen-3's P99 rose from 34.82 s to 67.86 s and remained 79.2% below HexGen-2's. The same request, request 8 in seed 997, set the control P99 for both systems and the original HexGen-2 P99. The cap restricts SGLang's KV preallocation capacity and disables two-way attention/FFN microbatching. The runtimes retain their own kernels, prefill schedulers, and queue management.

The three warmup requests matched all 25 reference tokens at every runtime start. The HexGen-2 per-request records show batch size one throughout. All 135 periodic HexGen-3 decode log entries show one active request. The captures also show GPUDirect NCCL for HexGen-2 and HexGen-3's FFN ranks, Mooncake on `mlx5_0`, and StepMesh RDMA. [Batch and transport verification](batch-verification.json)

Use the [batch-one commands](../../README.md#decode-batch-size-one-control) to rerun the control. The [paired results](paired-results.csv), [rate summary](rate-summary.csv), and [capture checksums](capture-sha256.txt) identify the recorded data.
