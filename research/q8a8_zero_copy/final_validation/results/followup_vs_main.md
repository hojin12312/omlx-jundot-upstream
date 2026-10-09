- session followup round 1: tree 0eac241e, 12 requests
- session followup round 2: tree 0eac241e, 12 requests
- session main round 1: tree 370d9442, 12 requests
- session main round 2: tree 370d9442, 12 requests

| L | n | main tok/s | follow-up tok/s | paired gain median (min..max) | TTFT main -> follow-up (s) | peak main / follow-up (GiB) | settled active main / follow-up (GiB) | phys main / follow-up (GiB) |
|---:|---:|---:|---:|---|---|---|---|---|
| 1024 | 6 | 1105 | 1258 | +13.8% (+12.6..+13.9) | 0.926 -> 0.814 | 31.06 / 29.65 | 29.358 / 27.942 | 30.18 / 28.77 |
| 4096 | 6 | 1243 | 1401 | +12.7% (+12.3..+12.8) | 3.294 -> 2.925 | 32.31 / 30.90 | 29.358 / 27.942 | 30.18 / 28.77 |
| 8192 | 6 | 1248 | 1389 | +11.2% (+10.4..+11.7) | 6.565 -> 5.898 | 32.57 / 31.15 | 29.358 / 27.942 | 30.18 / 28.77 |

- main round 1, first A8 request (L=1024): TTFT 1.071 s, prepare_builds 400, native calls 0, group-major calls 400, active delta 1,520,435,200 B
- main round 2, first A8 request (L=1024): TTFT 1.083 s, prepare_builds 400, native calls 0, group-major calls 400, active delta 1,520,435,200 B
- followup round 1, first A8 request (L=1024): TTFT 0.822 s, prepare_builds 0, native calls 400, group-major calls 0, active delta 0 B
- followup round 2, first A8 request (L=1024): TTFT 0.820 s, prepare_builds 0, native calls 400, group-major calls 0, active delta 0 B

Output text, same prompt: 24 pairs, 24 identical
