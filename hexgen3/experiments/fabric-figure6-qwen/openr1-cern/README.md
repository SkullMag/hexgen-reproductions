# OpenR1-Math on the CERN FABRIC slice

This is the first dataset in the scaled Figure 6 reproduction. The physical pool was one NVIDIA A40 and two NVIDIA A30s. FABRIC inventory showed both CERN A40s allocated at the start of this session. The target pool of two A40s and two A30s remains a separate run.

## Fixed inputs

- BF16 `Qwen/Qwen1.5-MoE-A2.7B-Chat` at `ec052fda178e241c7c443468d2fa1db6618996be`, using the checkpoint already verified on the three GPUs. The model file manifest identity was `5d322584d22c982c73a014afebfbc41072351fa35c837ecae667ae95310931f1`.
- OpenR1-Math-220k `default/train` at `e4e141ec9dea9f8326f4d347be56105859b2bd68`. The [workload manifest](workload-manifest.json) records the input shard hash, tokenizer hashes, and disjoint 32-request pilot and 480-request measured traces. The frozen Qwen chat token IDs and the same requested generation lengths go to both systems. Greedy generation ignores EOS so each request generates its assigned token budget. Output lengths are bounded to 64–256 tokens for the scaled run. [Dataset snapshot](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/tree/e4e141ec9dea9f8326f4d347be56105859b2bd68)
- HexGen-2 uses the CERN Qwen port with PyTorch 2.5.1 and Transformers 4.51.3. HexGen-3 uses the release at `efa3ed99e4337e5443f5fe4ee8c2fb2c29076de7`, SGLang 0.4.9.post6, PyTorch 2.7.1, Transformers 4.54.1, Triton 3.3.1, CUDA 12.6, Mooncake 0.3.7, and the released StepMesh C++ implementation compiled with GPUDirect support. [HexGen-3 release](https://github.com/Relaxed-System-Lab/HexGen-3/tree/efa3ed99e4337e5443f5fe4ee8c2fb2c29076de7)

## Placement

Both systems ran in the `hexgen2-qwen-cern-20261123` slice. The private data plane was `qwen-data`, a CERN L2Bridge on `10.86.6.0/24`. The GPU nodes had ConnectX-6 virtual functions at `mlx5_0:1`, RoCE v2 GID index 3, and NVIDIA driver 580.178.04.

| Host | Private IP | GPU | HexGen-2 | HexGen-3 |
| --- | --- | --- | --- | --- |
| `cern-w1` | `10.86.6.11` | A40 | Full prefill | Full prefill and HTTP router |
| `cern-w3` | `10.86.6.12` | A30 | Decode layers 0–11 | FFN tensor-parallel rank 0 |
| `cern-w6` | `10.86.6.13` | A30 | Decode layers 12–23 | Decode attention, FFN tensor-parallel rank 1, and StepMesh scheduler |
| `cern-w5` | `10.86.6.14` | CPU only | Load generator | Same load generator |

HexGen-2 caps batch size at two and uses GPUDirect NCCL for KV transfer and decoder pipeline traffic. HexGen-3 uses Mooncake/RDMA for KV transfer, StepMesh verbs for attention-to-FFN traffic across hosts, and NCCL GPUDirect for the two FFN tensor-parallel ranks. HexGen-3 uses two AFD microbatches where batch size permits, Triton attention, and eager execution with CUDA graphs, overlap scheduling, and radix caching disabled.

## Qualification

HexGen-2 passed the three frozen requests over the data plane from the CPU node and returned the same 25 tokens as the single-model reference. An unpatched HexGen-3 attempt returned the same three first tokens, then diverged during both multi-token requests. The released loader skipped the shared expert's merged gate and up projections in the FFN role. [The patch](qwen-shared-expert.patch) loads those parameters while retaining the role-specific attention loading. After applying the patch, HexGen-3 matched all 25 tokens from HexGen-2 on the same three requests. The [qualification receipt](qualification-results.json) records the matched token counts and the patched source checksum.

The StepMesh scheduler is launched externally before the model processes to provide a listening RDMA endpoint when FFN and attention initialize. The SGLang workers use `--skip-tokenizer-init` to return token IDs for the input IDs prepared on the CPU client. The qualification logs record the Mooncake interface and RoCE GID, StepMesh worker and server initialization, and FFN NCCL paths using `NET/IB/1/GDRDMA`.

## Commands

On each GPU node place the released HexGen-3 checkout in `~/hexgen3-openr1` and the baseline and pinned checkpoint in `~/hexgen2-qwen`. From the HexGen-3 checkout root, run `patch -p1 < qwen-shared-expert.patch`. Stage the launchers in their corresponding directories. Build HexGen-3 with `runtime3.sh`, `toolchain.sh`, and `stepmesh-build.sh`. Stage the replay client and traces in `~/openr1-controller` on the CPU node. The controller scripts take SSH wrapper paths. The GPU wrapper accepts one role from `prefill`, `decode-1`, or `decode-2` followed by an SSH command. The CPU wrapper accepts the SSH command.

```bash
bash switch-openr1.sh "$gpu_ssh" "$cpu_ssh" start hexgen2 h2-openr1-calibration 29667
bash replay-openr1.sh hexgen2 h2-opening-warmup finite --request-timeout 100
bash replay-openr1.sh hexgen2 h2-openr1-calibration offline --duration 205
PYTHONPATH=hexgen2/source .venv/bin/python freeze-schedules.py \
  runs/h2-openr1-calibration/hexgen2/summary.json schedules
python3 run-matrix.py plan.json captures --switch switch-openr1.sh \
  --gpu "$gpu_ssh" --cpu "$cpu_ssh" --pilot-runtime h2-openr1-calibration
python3 analyze-openr1.py plan.json captures analysis
```

The middle four lines starting with `bash replay-openr1.sh` run from the CPU node's experiment directory. Copy `schedules/plan.json` to the controller that holds the SSH wrappers before running the matrix. The runner reuses a drained runtime when the same method ends one pair and starts the next. A fresh runtime receives the three frozen warmup requests before measurement starts.

All online pairs use the same trace, arrival offsets, duration, request IDs, and client clock. The analysis rejects any mismatch. The throughput metric is completed requests divided by the fixed window duration. P99 is nearest-rank elapsed time from HTTP submission to the final token, over successes including the drain. Report successes, failures, and tail sample counts alongside P99. The physical pool was held fixed across arms, so relative requests per cost use the ratio of measured requests per second at the compared operating points.

## Frozen sweep

The four-request calibration queue ran for a 205-second measurement window. The first three successful completions arrived at 68.685, 137.273, and 205.515 seconds. Two requests timed out during the drain. The calibration uses the median 68.415-second gap between those three completions, which estimates 0.01462 requests per second while the decoder stays busy. The [calibration record](schedules/calibration.json) retains the fixed-window estimate as well. The controller stopped that runtime and started a fresh HexGen-2 instance before the first measured pair.

Seeds 61, 305, and 997 each cover five offered loads. The primary 0.85× load has 480-second windows and the other rates have 240-second windows. The [replay manifest](schedules/plan.json) records every requested rate, duration, offered count, and arm order. Each arm has the same 450-second HTTP request timeout and 460-second drain allowance. Small request counts constrain the resolution of empirical P99.
