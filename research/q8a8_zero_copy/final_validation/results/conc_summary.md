| session | scenario | wall s | requests ok / errors | peak GiB | quiet active GiB | quiet phys GiB |
|---|---|---:|---|---:|---:|---:|
| c1_native | pp_two_8192_prefills | 14.2 | 2 / 0 | 31.15 | 27.942 | 28.762 |
| c1_native | pd_prefill_during_decode | 14.1 | 2 / 0 | 31.15 | 27.942 | 28.764 |
| c1_native | cancel_during_prefill | 1.5 | 1 / 0 | 30.77 | 27.942 | 28.764 |
| c1_native | after_cancel_normal | 2.3 | 1 / 0 | 30.77 | 27.942 | 28.765 |
| c1_native | stagger_short_ends_first | 7.4 | 2 / 0 | 31.15 | 27.942 | 28.765 |
| c2_native | pp_two_8192_prefills | 14.3 | 2 / 0 | 31.42 | 27.942 | 28.768 |
| c2_native | pd_prefill_during_decode | 13.5 | 2 / 0 | 30.11 | 27.949 | 28.778 |
| c2_native | cancel_during_prefill | 1.5 | 1 / 0 | 30.77 | 27.949 | 28.778 |
| c2_native | after_cancel_normal | 2.3 | 1 / 0 | 30.77 | 27.949 | 28.778 |
| c2_native | stagger_short_ends_first | 7.5 | 2 / 0 | 31.16 | 27.949 | 28.779 |
| c2_native_mtp | pp_two_8192_prefills | 16.4 | 2 / 0 | 35.49 | 31.949 | 32.782 |
| c2_native_mtp | pd_prefill_during_decode | 11.8 | 2 / 0 | 35.19 | 31.949 | 32.785 |
| c2_native_mtp | cancel_during_prefill | 1.6 | 1 / 0 | 34.64 | 31.949 | 32.785 |
| c2_native_mtp | after_cancel_normal | 2.3 | 1 / 0 | 34.77 | 31.949 | 32.786 |
| c2_native_mtp | stagger_short_ends_first | 8.7 | 2 / 0 | 35.19 | 31.949 | 32.786 |
| c2_transposed | pp_two_8192_prefills | 15.1 | 2 / 0 | 32.83 | 29.358 | 30.186 |
| c2_transposed | pd_prefill_during_decode | 14.3 | 2 / 0 | 31.43 | 29.365 | 30.196 |
| c2_transposed | cancel_during_prefill | 1.5 | 1 / 0 | 32.15 | 29.365 | 30.196 |
| c2_transposed | after_cancel_normal | 2.5 | 1 / 0 | 32.19 | 29.365 | 30.196 |
| c2_transposed | stagger_short_ends_first | 8.2 | 2 / 0 | 32.57 | 29.365 | 30.197 |
| c2cache_native | pp_two_8192_prefills | 14.4 | 2 / 0 | 31.42 | 27.942 | 28.852 |
| c2cache_native | pd_prefill_during_decode | 13.6 | 2 / 0 | 30.16 | 27.949 | 28.887 |
| c2cache_native | cancel_during_prefill | 1.6 | 1 / 0 | 30.77 | 27.949 | 28.888 |
| c2cache_native | after_cancel_normal | 2.3 | 1 / 0 | 30.77 | 27.949 | 28.888 |
| c2cache_native | stagger_short_ends_first | 7.5 | 2 / 0 | 31.16 | 27.949 | 28.892 |
| c2cache_native | c4_four_4096_prefills | 15.9 | 4 / 0 | 31.49 | 27.949 | 28.877 |
| c2cache_native | c4_decode_plus_three_prefills | 17.8 | 4 / 0 | 31.28 | 27.949 | 28.878 |
| c2cache_native | cancel_during_cached_prefill | 1.0 | 1 / 0 | 29.99 | 27.949 | 28.879 |
| c2cache_native | same_prompt_after_cancel | 1.0 | 1 / 0 | 28.44 | 27.949 | 28.879 |
| c2cache_transposed | pp_two_8192_prefills | 15.3 | 2 / 0 | 33.43 | 29.358 | 30.262 |
| c2cache_transposed | pd_prefill_during_decode | 14.5 | 2 / 0 | 31.58 | 29.365 | 30.300 |
| c2cache_transposed | cancel_during_prefill | 1.5 | 1 / 0 | 32.15 | 29.365 | 30.300 |
| c2cache_transposed | after_cancel_normal | 2.5 | 1 / 0 | 32.19 | 29.365 | 30.300 |
| c2cache_transposed | stagger_short_ends_first | 8.4 | 2 / 0 | 32.57 | 29.365 | 30.303 |
| c2cache_transposed | c4_four_4096_prefills | 17.1 | 4 / 0 | 32.99 | 29.365 | 30.306 |
| c2cache_transposed | c4_decode_plus_three_prefills | 19.2 | 4 / 0 | 32.32 | 29.365 | 30.307 |
| c2cache_transposed | cancel_during_cached_prefill | 1.0 | 1 / 0 | 31.75 | 29.365 | 30.307 |
| c2cache_transposed | same_prompt_after_cancel | 1.0 | 1 / 0 | 29.86 | 29.365 | 30.307 |
| c4_native | pp_two_8192_prefills | 14.3 | 2 / 0 | 31.42 | 27.942 | 28.769 |
| c4_native | pd_prefill_during_decode | 13.6 | 2 / 0 | 30.11 | 27.949 | 28.779 |
| c4_native | cancel_during_prefill | 1.5 | 1 / 0 | 30.77 | 27.949 | 28.780 |
| c4_native | after_cancel_normal | 3.3 | 1 / 0 | 30.77 | 27.949 | 28.780 |
| c4_native | stagger_short_ends_first | 12.2 | 2 / 0 | 31.16 | 27.949 | 28.780 |
| c4_native | c4_four_4096_prefills | 24.1 | 4 / 0 | 31.44 | 27.949 | 28.782 |
| c4_native | c4_decode_plus_three_prefills | 28.1 | 4 / 0 | 31.61 | 27.949 | 28.783 |

| session | loop requests | errors | active min..max GiB | final active GiB | unload/reload cycles | active after unload | active after reload+request | suspect log lines |
|---|---:|---:|---|---:|---:|---|---|---:|
| c1_native | 20 | 0 | 27.942..27.942 | 27.942 | 0 | [] | [] | 0 |
| c2_native | 20 | 0 | 27.949..27.949 | 27.949 | 0 | [] | [] | 0 |
| c2_native_mtp | 20 | 0 | 31.949..31.949 | 31.949 | 0 | [] | [] | 0 |
| c2_transposed | 20 | 0 | 29.365..29.365 | 29.365 | 0 | [] | [] | 0 |
| c2cache_native | 20 | 0 | 27.942..27.942 | 27.942 | 2 | [0.0, 0.0] | [27.942, 27.942] | 0 |
| c2cache_transposed | 20 | 0 | 29.358..29.358 | 29.358 | 2 | [0.0, 0.0] | [29.358, 29.358] | 0 |
| c4_native | 20 | 0 | 27.942..27.942 | 27.942 | 2 | [0.0, 0.0] | [27.942, 27.942] | 0 |
