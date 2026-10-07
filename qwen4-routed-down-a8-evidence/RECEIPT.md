# Validation receipt: routed Down A8

## Pins

| Item | Value |
|---|---|
| Measured reference | `4b6b4458a53dfb1a3c6ce2af7379f75ce13e24bd` (`upstream/main` after #4320 and the maintainer's follow-up `726b8666`) |
| Measured candidate | `2d313a591fca71fe29a6746b70f0056d514e8a42`, two commits on the reference |
| Final branch | `d392aa1c4dae27f50eb26d171217b1052c22f2b6`, the same two commits rebased onto `2238a44477a91123051d0e15ac8b3b4e8a95fa83` and trimmed |
| Hardware / OS | Mac17,6, Apple M5 Max, 40 GPU cores, 128 GiB, macOS 27.0.1, AC power, no thermal warning |
| Toolchain | Python 3.11.15, mlx 0.32.3, mlx-lm 0.31.4.dev132+g94cdcae13, tokenizers 0.23.2 |
| Native libraries (sha256) | candidate `092ecdd2b4d1ac9cf65f1b570143370fb0fc948c77d0c50062b7419e9b24d759`, reference `a595bfb1453f6ebba47a3c493b7be7f1b810dba90568dc4e0ae29962f7ae6f29` |
| A8 module at the measured candidate (sha256) | `6add445caad7fc9badf109e96d25c7b16224921a18fdc055c04b7e459fdcc50b` |
| Checkpoint (`qwen3.8-flash-next-oQ4e-mtp`, `model_type qwen4_exp`) | `config.json` `319b334a…0824`, `tokenizer.json` `0997f410…29b9f3`, `model.safetensors.index.json` `05f70b01…73d2`. The local copy has no stored upstream revision id. |
| Corpus | `bundle/corpus.json`, 302,250 token ids, sha256 `fa11d794cea95de0aa88966e77602aa0615173304ae3b69004ac60d42ed2ab62` |

## Main campaign (`analysis/perf_main.txt`)

Two fresh-server sessions of six pairs per length (the second starts with the opposite condition), Down A16 vs A8, dense A8 and
Gate+Up A8 on in both, scheduler prefill memory guard off in both, `max_tokens=1`. Per pair: identical request bytes and prompt
ids (sha256), equal `usage.prompt_tokens`, `cached_tokens` reported as 0. One pair (32K, session 1) had an empty-text first token
in both conditions and was retried with a new nonce. There was no other invalid sample.

| Input | Down A16, tok/s | Down A8, tok/s | Paired gain, 95% t-CI | A8 faster | median TTFT A16 / A8 |
|---|---:|---:|---:|---:|---:|
| 16K | 2846.6 | 2899.7 | +1.92% [+0.73%, +3.11%] | 11/12 | 5.83 / 5.71 s |
| 32K | 2679.1 | 2733.5 | +2.03% [+1.88%, +2.19%] | 12/12 | 12.23 / 12.01 s |
| 64K | 2570.5 | 2620.0 | +1.93% [+1.75%, +2.10%] | 12/12 | 25.56 / 25.05 s |
| 128K | 2433.5 | 2480.4 | +1.93% [+1.76%, +2.09%] | 12/12 | 53.87 / 52.90 s |

Per session: 16K +1.77 / +2.07, 32K +2.02 / +2.05, 64K +1.83 / +2.02, 128K +1.92 / +1.94 %. The 16K row is the noisiest (one
pair at −3.48%). Down A8 results per request were 96 / 192 / 384 / 768 in the A8 condition and 0 in A16, matching the enqueued
`STAGED=2` Down launches. The largest Down call had 81,920 rows (8192-token chunk times top-10).

## Controls

- **A/A** (`analysis/aa_control.txt`): 32K −0.04% [−0.64%, +0.56%], 128K +0.01% [−0.15%, +0.17%]. The stop rule (a gain of 0.5% or
  more with an interval excluding 0) was not triggered.
- **Reference control** (`analysis/reference_control.txt`; reference build, candidate with Down off, candidate with Down on, as
  separate processes, two sessions with reversed order):

| Input | Reference, tok/s | Candidate, Down A8 off | Candidate, Down A8 on | off vs reference | on vs reference | triplets |
|---|---:|---:|---:|---:|---:|---:|
| 8K | 3118.7 | 3122.9 | 3188.5 | +0.13% [−0.15%, +0.42%] | +2.24% [+1.96%, +2.52%] | 6 |
| 32K | 2794.2 | 2804.8 | 2864.3 | +0.38% [+0.34%, +0.41%] | +2.51% [+2.15%, +2.86%] | 5 |

  For one 32K prompt the reference and the candidate with Down off produced an empty-text first token at two nonces, while the
  candidate with Down on did not, so the request bytes differ and that prompt has no matched triplet. This is a first-token
  difference caused by Down A8.
- **Instrumentation** (`analysis/instrumentation.txt`): without hook or counters, 32K on vs off is +2.53% [+1.82%, +3.25%]. With the
  hook it is +2.12% [+1.78%, +2.46%]. The intervals overlap and the point estimates differ by 0.41 percentage points.

## Memory (`analysis/memory.txt`)

MLX active-memory counters, chunk 8192, two measurements per condition and length with identical results.

| Input | Peak active, A16 / A8 (GiB) | Settled active, A16 / A8 (GiB) |
|---|---|---|
| 32K | 104.486 / 104.403 | 97.882 / 97.882 |
| 64K | 105.354 / 105.272 | 97.882 / 97.882 |
| 128K | 107.092 / 107.010 | 97.882 / 97.882 |

## Normal-guard serving smoke at 8K

With the scheduler memory guard on, a warm-up and five timed 8K requests completed with one token each, and Down A8 ran on every
request. Requests were 5 s apart. A first attempt with back-to-back requests on an earlier base had one request rejected by the
scheduler's preflight guard.

## Quality (`analysis/quality.txt`)

The full 36-sequence corpus (24 primary and 12 structured sequences, 163,840 positions) ran on the measured candidate with 4096-token
chunks. All 648 per-position arrays are bit-identical to an earlier run of the same corpus. The tables of the PR body come from this
analysis.

## Tool-call smoke

Ten long tool-calling chat requests (1.5K to 12K tokens), each sent with Down A16 and Down A8 in one server. Schema-valid tool calls:
10/10 in both. Same tool and arguments: 8/10. The expected tool was not chosen in the same two cases under both conditions. Down A8
results were counted only in the A8 condition.

## Tests and static checks

| Check | Result |
|---|---|
| Native targeted, measured candidate | 247 passed, 0 skipped |
| Native targeted, first commit of the final branch | 206 passed, 0 skipped (the second commit adds the 13 dispatch tests). Ruff and Black checks of the first commit are clean as well (`analysis/final_branch_checks.txt`). |
| Native targeted, final branch | 219 passed, 0 skipped. The count is lower because the final branch removes tests that repeat upstream Gate+Up coverage and merges overlapping Down tests. Each test of `test_m5_gather_qmm_a8.py` also passes in its own process. |
| Native default suite, measured candidate | 17,433 passed, 384 skipped, 5 failed. The same 5 DeepSeek-V4 DSpark ring tests fail with the same assertion on the measured reference (17,376 passed), built with its own native libraries (`analysis/native_default_failures_vs_reference.txt`). Not repeated on the final branch. |
| CI-equivalent, measured candidate | 16,978 passed, 844 skipped, 0 failed (reference: 16,970 passed, 795 skipped). Node.js was not on `PATH` in that run, so about 126 Node.js tests skipped. |
| CI-equivalent, final branch | 17,155 passed, 690 skipped, 0 failed. Node.js was on `PATH`, so this run is not comparable with the measured-candidate run above (about 126 Node.js tests moved from skipped to passed). No CI-equivalent run was made on the final base without this change. Kernel-backed tests skip without native kernels and NAX hardware (56 of the 74 tests in `test_m5_gather_qmm_a8.py`). |
| Static checks, final branch | Ruff 123 findings on base and head, none new. Black: no finding on a changed line. `git diff --check`: clean. |

CI-equivalent means a fresh Python 3.11 environment, `pip install -e ".[mcp]"` plus pytest, pytest-asyncio and pytest-xdist, and
`pytest tests/ -m "not slow and not integration" -n 3 --dist loadgroup`, with no native build.

## Rebase and cleanup after the measurements

Between the measured reference and the final base, `upstream/main` gained #4335 (fused MoE decode for FP16 and top-8 routing) and #4333
(EmbeddingGemma 2). Neither touches a file of this change or the A8 prefill path. `qwen35_moe_router.py` and
`qwen35_moe_routed_decode.py` only handle decode and verify widths. The rebase had no conflicts, and `git range-diff` shows the two
commits as unchanged before the cleanup.

The cleanup removed comments and tests that repeat upstream coverage, renamed the `G = 10` canary and its cache key
(`_self_test_g10`, `"plain-g10"`), restricted the Metal `static_assert` to `SG == 10`, removed the small-input guard in `supports()`
(the routed call sites require at least 128 tokens), and moved the UI description change out of the PR. A comparison of the Python AST
without docstrings between the measured candidate and the final branch shows only the guard removal and the renames. The Metal source
without comments differs only in the `static_assert`. No benchmark, quality, memory or tool-call run was repeated on the final branch.

## Test scope

The Down-specific tests cover the G = 10 kernel (stage-mode selection, launch template parameters, staged-vs-scalar bit equality in BF16 and
FP16, the model's E = 512 shape, canary success and failure, canary independence from Gate+Up) and the Down dispatch (both call paths,
unmeasured geometry, Down exception, failed Down canary, the Down kill switch and the global kill switch). The default-on behavior and
the Down kill switch are checked inside the two eligible-path tests (default-on in the first half, `OMLX_M5_ROUTED_DOWN_A8=0` in the
second half), which clear the variable from the environment first. Short sequences, batched decode, MTP exclusion, expert offload and
module opt-in are not repeated for Down because Down only runs after a successful Gate+Up, and the existing Gate+Up tests cover them.
