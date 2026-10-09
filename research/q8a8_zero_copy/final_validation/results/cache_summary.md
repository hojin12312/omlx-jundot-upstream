| step | arm | prompt tok | cached | TTFT s | Q8 A8 rows per call (calls) | native/transposed calls | peak GiB | settled GiB | prepared layers | text sha |
|---|---|---:|---:|---:|---|---|---:|---:|---:|---|
| L4096.miss | off | 4096 | 0 | 4.305 | - | 0/0 | 30.86 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.miss | transposed | 4096 | 0 | 3.488 | {'2047': 400, '2048': 400} | 0/800 | 32.31 | 29.36 | 400 | 74dc1e04d5c21512 |
| L4096.miss | native | 4096 | 0 | 2.974 | {'2047': 400, '2048': 400} | 800/0 | 30.90 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.miss | transposed_mtp | 4096 | 0 | 3.563 | {'2047': 400, '2048': 400} | 0/800 | 36.35 | 33.38 | 400 | 74dc1e04d5c21512 |
| L4096.miss | native_mtp | 4096 | 0 | 3.044 | {'2047': 400, '2048': 400} | 800/0 | 34.93 | 31.97 | 0 | 74dc1e04d5c21512 |
| L4096.hit_identical | off | 4096 | 4095 | 0.158 | - | 0/0 | 28.43 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.hit_identical | transposed | 4096 | 4095 | 0.162 | - | 0/0 | 29.85 | 29.36 | 400 | 74dc1e04d5c21512 |
| L4096.hit_identical | native | 4096 | 4095 | 0.167 | - | 0/0 | 28.43 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.hit_identical | transposed_mtp | 4096 | 4095 | 0.173 | - | 0/0 | 34.28 | 33.38 | 400 | 74dc1e04d5c21512 |
| L4096.hit_identical | native_mtp | 4096 | 4095 | 0.172 | - | 0/0 | 32.86 | 31.97 | 0 | 74dc1e04d5c21512 |
| L4096.partial_half | off | 4096 | 0 | 4.302 | - | 0/0 | 30.86 | 27.94 | 0 | 8435e3ae1236ee56 |
| L4096.partial_half | transposed | 4096 | 0 | 3.317 | {'2047': 400, '2048': 400} | 0/800 | 32.31 | 29.36 | 400 | 8435e3ae1236ee56 |
| L4096.partial_half | native | 4096 | 0 | 2.956 | {'2047': 400, '2048': 400} | 800/0 | 30.90 | 27.94 | 0 | 8435e3ae1236ee56 |
| L4096.partial_half | transposed_mtp | 4096 | 0 | 3.392 | {'2047': 400, '2048': 400} | 0/800 | 36.37 | 33.40 | 400 | 8435e3ae1236ee56 |
| L4096.partial_half | native_mtp | 4096 | 0 | 3.032 | {'2047': 400, '2048': 400} | 800/0 | 34.95 | 31.98 | 0 | 8435e3ae1236ee56 |
| L4096.hit_prefix+suffix | off | 5117 | 2048 | 3.383 | - | 0/0 | 30.93 | 27.94 | 0 | befd846049f3ebc8 |
| L4096.hit_prefix+suffix | transposed | 5117 | 2048 | 2.611 | {'1020': 400, '2048': 400} | 0/800 | 32.38 | 29.36 | 400 | befd846049f3ebc8 |
| L4096.hit_prefix+suffix | native | 5117 | 2048 | 2.320 | {'1020': 400, '2048': 400} | 800/0 | 30.96 | 27.94 | 0 | befd846049f3ebc8 |
| L4096.hit_prefix+suffix | transposed_mtp | 5117 | 2048 | 2.666 | {'1020': 400, '2048': 400} | 0/800 | 36.42 | 33.40 | 400 | befd846049f3ebc8 |
| L4096.hit_prefix+suffix | native_mtp | 5117 | 2048 | 2.378 | {'1020': 400, '2048': 400} | 800/0 | 35.00 | 31.98 | 0 | befd846049f3ebc8 |
| L4096.hit_suffix100 | off | 4193 | 4096 | 0.366 | - | 0/0 | 28.59 | 27.94 | 0 | 668b6b61b065ace9 |
| L4096.hit_suffix100 | transposed | 4193 | 4096 | 0.366 | - | 0/0 | 30.01 | 29.36 | 400 | 668b6b61b065ace9 |
| L4096.hit_suffix100 | native | 4193 | 4096 | 0.365 | - | 0/0 | 28.59 | 27.94 | 0 | 668b6b61b065ace9 |
| L4096.hit_suffix100 | transposed_mtp | 4193 | 4096 | 0.381 | - | 0/0 | 34.28 | 33.40 | 400 | 668b6b61b065ace9 |
| L4096.hit_suffix100 | native_mtp | 4193 | 4096 | 0.378 | - | 0/0 | 32.86 | 31.98 | 0 | 668b6b61b065ace9 |
| L4096.hit_suffix130 | off | 4223 | 4192 | 0.311 | - | 0/0 | 28.50 | 27.94 | 0 | b78dc80b55838ff0 |
| L4096.hit_suffix130 | transposed | 4223 | 4192 | 0.314 | - | 0/0 | 29.92 | 29.36 | 400 | b78dc80b55838ff0 |
| L4096.hit_suffix130 | native | 4223 | 4192 | 0.316 | - | 0/0 | 28.50 | 27.94 | 0 | b78dc80b55838ff0 |
| L4096.hit_suffix130 | transposed_mtp | 4223 | 4192 | 0.325 | - | 0/0 | 34.28 | 33.40 | 400 | b78dc80b55838ff0 |
| L4096.hit_suffix130 | native_mtp | 4223 | 4192 | 0.325 | - | 0/0 | 32.86 | 31.98 | 0 | b78dc80b55838ff0 |
| L4096.after_wipe_reload | off | 4096 | 0 | 4.503 | - | 0/0 | 30.86 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.after_wipe_reload | transposed | 4096 | 0 | 3.650 | {'2047': 400, '2048': 400} | 0/800 | 32.31 | 29.36 | 400 | 74dc1e04d5c21512 |
| L4096.after_wipe_reload | native | 4096 | 0 | 3.161 | {'2047': 400, '2048': 400} | 800/0 | 30.90 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.after_wipe_reload | transposed_mtp | 4096 | 0 | 3.795 | {'2047': 400, '2048': 400} | 0/800 | 34.28 | 33.38 | 400 | 74dc1e04d5c21512 |
| L4096.after_wipe_reload | native_mtp | 4096 | 0 | 3.318 | {'2047': 400, '2048': 400} | 800/0 | 32.86 | 31.97 | 0 | 74dc1e04d5c21512 |
| L4096.hit_after_reload | off | 4096 | 4095 | 0.170 | - | 0/0 | 28.43 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.hit_after_reload | transposed | 4096 | 4095 | 0.160 | - | 0/0 | 29.85 | 29.36 | 400 | 74dc1e04d5c21512 |
| L4096.hit_after_reload | native | 4096 | 4095 | 0.160 | - | 0/0 | 28.43 | 27.94 | 0 | 74dc1e04d5c21512 |
| L4096.hit_after_reload | transposed_mtp | 4096 | 4095 | 0.173 | - | 0/0 | 34.28 | 33.38 | 400 | 74dc1e04d5c21512 |
| L4096.hit_after_reload | native_mtp | 4096 | 4095 | 0.170 | - | 0/0 | 32.86 | 31.97 | 0 | 74dc1e04d5c21512 |
| L8192.miss | off | 8192 | 0 | 8.826 | - | 0/0 | 31.11 | 27.94 | 0 | 1e49ba5ded5aabee |
| L8192.miss | transposed | 8192 | 0 | 6.603 | {'2047': 400, '2048': 1200} | 0/1600 | 32.57 | 29.36 | 400 | 20f31806dfe7f15f |
| L8192.miss | native | 8192 | 0 | 5.982 | {'2047': 400, '2048': 1200} | 1600/0 | 31.15 | 27.94 | 0 | 20f31806dfe7f15f |
| L8192.hit_identical | off | 8192 | 8191 | 0.181 | - | 0/0 | 28.74 | 27.94 | 0 | 1e49ba5ded5aabee |
| L8192.hit_identical | transposed | 8192 | 8191 | 0.181 | - | 0/0 | 30.16 | 29.36 | 400 | 20f31806dfe7f15f |
| L8192.hit_identical | native | 8192 | 8191 | 0.179 | - | 0/0 | 28.74 | 27.94 | 0 | 20f31806dfe7f15f |
| L8192.partial_half | off | 8192 | 2048 | 6.621 | - | 0/0 | 31.11 | 27.94 | 0 | 007bc62ec852c52c |
| L8192.partial_half | transposed | 8192 | 2048 | 5.072 | {'2047': 400, '2048': 800} | 0/1200 | 32.57 | 29.36 | 400 | 007bc62ec852c52c |
| L8192.partial_half | native | 8192 | 2048 | 4.561 | {'2047': 400, '2048': 800} | 1200/0 | 31.15 | 27.94 | 0 | 007bc62ec852c52c |
| L8192.hit_prefix+suffix | off | 10238 | 6144 | 4.666 | - | 0/0 | 31.24 | 27.94 | 0 | f3ed13044c1e4fb7 |
| L8192.hit_prefix+suffix | transposed | 10238 | 6144 | 3.592 | {'2045': 400, '2048': 400} | 0/800 | 32.69 | 29.36 | 400 | f3ed13044c1e4fb7 |
| L8192.hit_prefix+suffix | native | 10238 | 6144 | 3.242 | {'2045': 400, '2048': 400} | 800/0 | 31.27 | 27.94 | 0 | f3ed13044c1e4fb7 |
| L8192.hit_suffix100 | off | 8290 | 8192 | 0.402 | - | 0/0 | 28.81 | 27.94 | 0 | c79988a868d19299 |
| L8192.hit_suffix100 | transposed | 8290 | 8192 | 0.411 | - | 0/0 | 30.23 | 29.36 | 400 | c79988a868d19299 |
| L8192.hit_suffix100 | native | 8290 | 8192 | 0.408 | - | 0/0 | 28.81 | 27.94 | 0 | c79988a868d19299 |
| L8192.hit_suffix130 | off | 8320 | 8289 | 0.354 | - | 0/0 | 28.75 | 27.94 | 0 | c64b0bd2eff9e186 |
| L8192.hit_suffix130 | transposed | 8320 | 8289 | 0.354 | - | 0/0 | 30.17 | 29.36 | 400 | c64b0bd2eff9e186 |
| L8192.hit_suffix130 | native | 8320 | 8289 | 0.355 | - | 0/0 | 28.75 | 27.94 | 0 | c64b0bd2eff9e186 |
| L8192.after_wipe_reload | off | 8192 | 0 | 9.078 | - | 0/0 | 31.11 | 27.94 | 0 | 1e49ba5ded5aabee |
| L8192.after_wipe_reload | transposed | 8192 | 0 | 6.925 | {'2047': 400, '2048': 1200} | 0/1600 | 32.57 | 29.36 | 400 | 20f31806dfe7f15f |
| L8192.after_wipe_reload | native | 8192 | 0 | 6.151 | {'2047': 400, '2048': 1200} | 1600/0 | 31.15 | 27.94 | 0 | 20f31806dfe7f15f |
| L8192.hit_after_reload | off | 8192 | 8191 | 0.194 | - | 0/0 | 28.74 | 27.94 | 0 | 1e49ba5ded5aabee |
| L8192.hit_after_reload | transposed | 8192 | 8191 | 0.184 | - | 0/0 | 30.16 | 29.36 | 400 | 20f31806dfe7f15f |
| L8192.hit_after_reload | native | 8192 | 8191 | 0.182 | - | 0/0 | 28.74 | 27.94 | 0 | 20f31806dfe7f15f |

native vs transposed: 16 steps, text/cached/prompt mismatches: none
native_mtp vs transposed_mtp: 8 steps, text/cached/prompt mismatches: none
native max prepared layers 0 suspect log lines: 4
native_mtp max prepared layers 0 suspect log lines: 3
off max prepared layers 0 suspect log lines: 1
transposed max prepared layers 400 suspect log lines: 5
transposed_mtp max prepared layers 400 suspect log lines: 4
