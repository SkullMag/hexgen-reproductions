# Matched OpenR1 results

The static CERN comparison completed 15 paired replays across five offered rates and three seeds. Both arms used BF16 Qwen1.5-MoE-A2.7B-Chat on one A40 and two A30s, with the same separate CPU load generator. The target two-A40 configuration was unavailable at the inventory check.

## Primary operating point

The 0.85× load was 0.01242418 requests/s. Each replay used a 480-second window.

| Runtime | Finished within the windows | Finished after drain | Completed requests/min | P99 (s) |
| --- | ---: | ---: | ---: | ---: |
| HexGen-2 | 18 / 27 | 27 / 27 | 0.750 | 339.78 |
| HexGen-3 | 27 / 27 | 27 / 27 | 1.125 | 34.82 |

HexGen-3 completed 50% more requests inside the windows. Its observed P99 was 89.8% lower. Each P99 is the maximum of 27 successful requests. The shared pool makes the throughput ratio equal to the relative requests-per-cost ratio under a shared hourly rate.

## Full load sweep

| Load | Offered per arm | HexGen-2 in window | HexGen-3 in window | HexGen-2 P99 (s) | HexGen-3 P99 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.50× | 8 | 5 | 7 | 138.91 | 40.83 |
| 0.85× | 27 | 18 | 27 | 339.78 | 34.82 |
| 1.00× | 17 | 7 | 14 | 365.08 | 34.32 |
| 1.10× | 18 | 8 | 16 | 374.92 | 34.34 |
| 1.40× | 21 | 10 | 20 | 419.16 | 35.58 |

Each system completed all 91 scheduled requests after drain with zero request failures. These 91 arrivals used 27 distinct prompts across the three seeds. All pairs verified identical arrival hashes, trace hashes, request IDs, token budgets, and the planned arm order. The largest per-run P99 submission lag was under 95 ms. P99 is pooled within each load across the three seeds. Both runtimes use the tested static placements and their own batching and inference kernels.

The repaired HexGen-3 loader matched all 25 smoke-test output tokens. Across the longer workload requests, the first token matched in 91 of 91 paired generations and the full sequence matched in 7 of 91. Every measured request generated its frozen token budget.

[Figure 6-style primary comparison](figure6-normalized-primary.png) · [Load sweep figure](openr1-figure6.png) · [Per-pair results](paired-results.csv) · [Rate summary](rate-summary.csv)

## What drove the HexGen-2 tail latency

The local Qwen adapter grouped requests only when both prompt length and generation budget matched. All 91 measured HexGen-2 requests decoded at batch size one, and the worker completed each decode batch before starting the next. Median decode service was 68.85 seconds for a 256-token output, about 3.7 tokens/s. Median prefill took 0.431 seconds, and the decode head's median KV receive interval was 5.5 ms.

The slowest primary-load request was request 8 in seed 997. It took 339.78 seconds end to end and 69.40 seconds in decode. Monotonic timestamps on the corresponding workers show 131.0 seconds before prefill began and another 68.8 seconds between KV receipt and decode start. A serial replay using the original arrival offsets and the measured decode durations reproduces 339.23 seconds for that request. The tail reflects the local adapter's serialized service under a burst of arrivals.

The port uses an eager Transformers MoE path that checks every expert per layer, and the worker synchronizes the CUDA stream at each token. These are candidates for GPU profiling. The measured comparison includes those runtime differences alongside attention and FFN disaggregation. [Qwen stage](../../../../../hexgen2/source/hexgen2/qwen_stage.py) · [Online worker](../../../../../hexgen2/source/hexgen2/comparison_worker.py)

The three primary OpenR1 pairs were rerun with both systems capped at one active decode request. [Batch size one results](batch-one/README.md)

## Higher-load probes

Three additional matched pairs used seed 61, 180-second windows, and the same physical placement and checkpoint. These single-seed probes report in-window completions and timeout outcomes.

| Offered requests/s | Offered per arm | HexGen-2 in window | HexGen-3 in window | HexGen-2 timeouts | HexGen-3 timeouts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05847 | 12 | 4 | 12 | 7 | 0 |
| 0.17540 | 36 | 4 | 28 | 32 | 0 |
| 0.35080 | 67 | 4 | 45 | 63 | 0 |

HexGen-2 completed four requests in each window, or 0.0222 requests/s. At the highest offered load, HexGen-3 completed 45 requests inside the window, or 0.2500 requests/s, and finished all 67 after drain. Its P99 at that point was 85.06 seconds and 22 completions fell outside the window. HexGen-3 had no request failures across the 115 extension arrivals. HexGen-2's extension failures were HTTP client timeouts after the measurement windows.

All 36 arm captures were saved and verified against SHA-256. The run controller stopped the benchmark processes at the end, and GPU checks showed zero allocated memory on all three workers.

[Overload per-pair results](overload-paired-results.csv) · [Overload rate summary](overload-rate-summary.csv) · [Capture hashes](capture-sha256.txt)
