# HexGen-2 versus HexGen-3 on FABRIC

This protocol scales the heterogeneous comparison in [HexGen-3 Figure 6](https://openreview.net/pdf?id=fVG57FKBNp) to Qwen1.5-MoE and a smaller GPU pool. The measurements cover static provisioning with autoscaling disabled. The paper reports a 20.3% mean and 25.8% maximum increase in requests per dollar for HexGen-3 over HexGen-2. Its latency-cost bars imply an approximately 14% mean reduction relative to HexGen-2. These values are references for the scaled comparison.

## Inputs and placement

| Input | Planned value |
| --- | --- |
| Model and tokenizer | [`Qwen/Qwen1.5-MoE-A2.7B-Chat` at `ec052fd`](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat/tree/ec052fda178e241c7c443468d2fa1db6618996be), BF16 for both systems |
| HexGen-2 starting point | [Native CERN runtime at `1bd498b`](../../../hexgen2/experiments/cern-gpudirect-20261109/original13b/runtime/) and the Qwen port described below. |
| HexGen-3 starting point | [Released runtime at `efa3ed9`](https://github.com/Relaxed-System-Lab/HexGen-3/tree/efa3ed99e4337e5443f5fe4ee8c2fb2c29076de7) |
| FABRIC slice | Target 2 A40s and 2 A30s with qualified GPUDirect RDMA links and a separate CPU load generator |
| HexGen-2 smoke test | 1 A40 for prefill and 2 A30s for decode, reusing the CERN roles. Start with Qwen layers `[0, 12)` and `[12, 24)` on the two decode stages. |
| HexGen-3 smoke test | 1 A40 for prefill, 1 A40 for FFN, and 2 A30s for decode attention |

Both systems get the full four-GPU pool in measured runs. Profile and select a static placement for each using pilot traffic, then record it before measured replay. Keep the physical GPUs and links fixed across paired runs. The smoke-test layouts are starting points for profiling.

## Bring-up order

1. Recover or port the original FABRIC provisioning and staging helpers. Reuse the [CERN launcher and network checks](../../../hexgen2/experiments/cern-gpudirect-20261109/controller/) to validate GPUDirect on the new reservation. Record the actual GPU-to-rank and NIC mapping, software versions, and measured link bandwidth. Confirm that the HexGen-3 attention-to-FFN path works over the reserved links.
2. Qualify **HexGen-2 with Qwen first**. Use an authors' Qwen-compatible build if available. Otherwise, port the CERN runtime's model-loading, inference, and KV transfer paths to Qwen and add online request admission. Compare a small set of fixed prompts against a single-model reference and inspect that prefill and decode execute on separate workers. The existing CERN runner provides an offline sanity check with OpenLLaMA-13B.
3. Qualify **HexGen-3 with the same checkpoint**. Run the same small queue, audit output lengths and correctness, and check that both the prefill-to-decode KV transfer and decode attention-to-FFN transfers occur. Use the same client, prompt tokenization, and generation settings in both arms.

## Measured runs

Use WildGPT, OpenThoughts, OpenR1-Math, and NuminaMath. Use the authors' samples and rates if available. Otherwise freeze each dataset revision, request IDs, tokenization, generation lengths, and seeded Poisson arrival offsets before the measured runs. Each paired run replays the same open-loop arrival schedule and payloads against the two systems. Use separate pilot requests for tuning.

The fallback load sweep is `0.50`, `0.85`, `1.00`, `1.10`, and `1.40` times the pilot estimate of sustainable HexGen-2 requests per second for each workload. Use three independent replay seeds per cell and share each seed across arms. Alternate the order of the systems across pairs. The fallback matrix has 2 systems × 4 workloads × 5 rates × 3 seeds = 120 runs, excluding qualification and pilots.

## Qwen qualification and replay code

The Qwen port uses Transformers 4.51.3 Qwen2-MoE layers with PyTorch 2.5.1 SDPA and the HexGen-2 reproduction's stage and cache-transfer adapter. Report this implementation choice with the results. The three-rank qualification module is [`hexgen2.qualify_qwen`](../../../hexgen2/source/hexgen2/qualify_qwen.py). It checks the generated tokens from a full prefill on rank 0 and transferred-cache decode on ranks 1 and 2 against colocated generation. The [four-rank online worker](../../../hexgen2/source/hexgen2/comparison_worker.py) loads the Qwen port when the checkpoint declares `qwen2_moe`. Its current prefill and decode replicas each use two pipeline stages.

Set `PYTHONPATH=hexgen2/source` from the repository root for these modules. Stage the pinned checkpoint on each worker with `hexgen2.fetch_model`. The qualification module takes `--model-dir`, `--trace`, and `--output`; run one process per rank with a shared rendezvous address and port. Each trace row contains `id`, `input_ids`, and `max_new_tokens`. Use [the three frozen qualification requests](qualification.jsonl) for the first smoke test. The pinned tokenizer applied its chat template with a generation prompt. The CUDA path uses NCCL and the rank-local device selected by `LOCAL_RANK`.

The [replay client](../../../hexgen2/source/hexgen2/comparison_client.py) takes `--request-format hexgen2` for the baseline and `--request-format sglang` for the HexGen-3 load balancer. In online mode, create and save a schedule with `--poisson-rate` and `--arrival-seed`, then give the other arm the saved `arrivals.json` through `--arrivals` with the same trace and duration. Set the client in-flight limit high enough to avoid throttling the open-loop schedule. Use the trace and arrival hashes in each summary to check the pair, and retain the recorded submission lag.

## Analysis

- **Throughput.** Measure completed requests per second during the steady window. Use the highest sustainable measured rate per workload for the throughput panel when paper settings are unavailable. Record completion rate, failures, and queue growth to identify overload. If either system stays stable at the highest offered rate, extend the sweep for both arms to bracket saturation.
- **P99 latency.** Measure from client submission to the final token at the same absolute offered rate for both arms. Use the paper's operating point if available, or the `0.85` point as the primary comparison. Report the full load sweep and error and timeout counts alongside the paired P99 values.
- **Cost.** Report raw throughput and latency, plus paired percentage differences. Add the paper's cost axes when its normalization and a common GPU rate are established. Use completed requests per second × 3600 ÷ hourly GPU cost for requests per dollar.

## Run record

Commit the tested launchers and resolved run configuration before collecting results. Record both runtime commit IDs, model and tokenizer revision, dependency versions, ranks and placements, dataset revisions, workload hashes, arrival seeds and rates, generation settings, warmup and measurement windows, and the criteria for identifying stable runs. Keep a per-run request log with request ID, submission and completion timestamps, token counts, status, and run configuration ID. Publish the aggregate table, paired plots, and the reporting command with the measured commit ID. Keep credentials, model weights, and large telemetry outside Git.

Local checks passed for BF16 staged logits against a full Qwen2-MoE fixture, exact token parity across three CPU Gloo ranks after transferring the cache, and paired replay against mock HTTP endpoints. HexGen-2 also passed full-checkpoint CUDA qualification on three CERN workers. HexGen-3 also passed the same three requests on the CERN slice after the Qwen FFN loader patch. The first matched Figure 6 workload used OpenR1-Math on one A40 and two A30s. The target four-GPU comparison remains pending. [CERN qualification run](cern-smoke-20261123/) · [OpenR1-Math run record](openr1-cern/) · [Measured results](openr1-cern/results/)
