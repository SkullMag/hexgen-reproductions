# Matched OpenR1 results

The first 0.85× pair used seed 61, 0.01242418 requests/s, and 480-second windows. Requests and arrival offsets match across the two client logs.

| Runtime | Finished within 480 s | Finished after drain | Completed requests/min | P99 (s) |
| --- | ---: | ---: | ---: | ---: |
| HexGen-2 | 5 / 7 | 7 / 7 | 0.625 | 77.30 |
| HexGen-3 | 7 / 7 | 7 / 7 | 0.875 | 27.79 |

HexGen-3 completed 40% more requests in the measurement window and its observed P99 was 64% lower. Each P99 is the maximum of seven successful requests. Both arms used one A40, two A30s, and the same CPU load generator. Seed 61 completed all five offered loads. Across that seed, each system completed all 21 offered requests after the drain. The highest rate was 0.02046 requests/s. During that window, HexGen-2 finished four of five and HexGen-3 finished five of five. Their P99 latencies were 99.52 and 28.04 seconds, respectively. Further seeds and a capacity bracket remain pending.

[Per-pair results](paired-results.csv) · [Rate summary](rate-summary.csv)
