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
