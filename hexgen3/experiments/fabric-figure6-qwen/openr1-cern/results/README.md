# Matched OpenR1 results

The primary 0.85× load was 0.01242418 requests/s. All three paired seeds completed with matching input IDs, arrival offsets, and alternating arm order. Each arm had a 480-second window per seed.

| Runtime | Finished within the windows | Finished after drain | Completed requests/min | P99 (s) |
| --- | ---: | ---: | ---: | ---: |
| HexGen-2 | 18 / 27 | 27 / 27 | 0.750 | 339.78 |
| HexGen-3 | 27 / 27 | 27 / 27 | 1.125 | 34.82 |

HexGen-3 completed 50% more requests inside the windows. Its observed P99 was 89.8% lower. Each P99 is the maximum of 27 successful requests. Both arms used one A40, two A30s, and the same CPU load generator. The shared pool makes the paired throughput ratio equal to the relative requests-per-cost ratio under a shared hourly rate. The remaining offered loads and overload probes measure behavior beyond this primary point.

[Normalized primary comparison](figure6-normalized-primary.png) · [Per-pair results](paired-results.csv) · [Rate summary](rate-summary.csv)
