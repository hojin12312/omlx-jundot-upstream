## Summary

Runs Q8 / GS64 / affine prefill projections of Qwen3.8 as W8A8 through the existing oQ A8 path on M5. Measured on `Jundot/Qwen3.8-27B-oQ8e-mtp`, prefill throughput rises by about 30–37% from 1K to 32K tokens.

The existing `qwen35_oq_a8_enabled` setting now also covers Q8; no new setting is added. With the setting off, behavior is unchanged. Users who already enable A8 for Q4/Q5 get Q8 A8 on checkpoints that contain Q8 projections.

The cost is about +1.4 GiB of persistent memory on this 30 GB checkpoint (see Memory), and a change in the output distribution that is concentrated in structured and tool-like text (see Correctness).

## Approach

- A new Metal kernel reads the checkpoint's packed Q8 weights in place. Flipping the top bit of each byte gives `q - 128` as INT8, and the per-group affine correction `s * (acc + 128 * r) + b * r` is applied in FP32, where `acc` is the INT32 tensor-op result and `r` is the activation group sum that Stage A already produces. No unpacked weight copy exists.
- Q8 uses a natural Stage A layout: activations stay in checkpoint K order, so no permuting copy is made. Q4/Q5 keep Stage A v8 and their kernels are unchanged.
- Q8 uses GS64 activation scaling, which keeps activation outliers from setting the scale of a whole row.
- MLP gate and up share one Stage A and down quantizes its own input, as for Q4/Q5. Mixed Stage-A layouts decline instead of sharing.
- In `qwen35_q4_mlp.py`, 8-bit prefill rows are offered to the A8 backend first, including below the existing 16384-row threshold of the Q8 A16 tile. When the backend declines, stock MLX runs; the A16 tile path is kept as it was.

## Scope

- Included: dense Q8 / GS64 / affine projections, prefill only, M5 with the native extension built.
- Unchanged: Q4/Q5 A8 kernels and routing, Q6, GS128, decode and target verify, and the Q8 A16 path. `lm_head` is not specially handled.
- Q8 MTP draft-head projections remain on their existing path; Q8A8 is scoped to target-model prefill.
- Flash-Next routed expert Down A8 (#4337) is separate. This PR does not depend on it.
- No UI change. The Q4/Q5 wording of the A8 description is left for a follow-up.
- Overlaps `omlx/patches/qwen35_oq_a8.py` with #4181; whichever lands second needs a small rebase in `_tag_modules` and `apply_qwen35_oq_a8_patch`.

## Performance

M5 Max, 40 GPU cores, 128 GiB; MLX 0.32.3; `Jundot/Qwen3.8-27B-oQ8e-mtp` at revision `c99e5aad8a478f71c10b9a3dde6709158b690da6`; Q8 / GS64 / affine; prefill chunk 2048; concurrency 1; prefix cache and MTP off; one streamed token.

Both conditions run in one server process; the same request payload is sent to each in rotating order, with a settle period before every request to limit thermal carry-over (10 s at 1K, 45 s at 4K–16K, 60 s at 32K). Throughput is prompt tokens divided by client-observed TTFT. Four paired tuples per length, and Q8A8 was faster in every pair.

| Input | A8 off, tok/s | Q8A8, tok/s | Paired gain, median (min–max) | TTFT off → on, s |
|---|---:|---:|---:|---|
| 1K | 846 | 1105 | +30.6% (+30.1 – +31.6) | 1.21 → 0.93 |
| 4K | 959 | 1246 | +29.9% (+29.7 – +30.1) | 4.27 → 3.29 |
| 8K | 930 | 1249 | +34.4% (+33.7 – +35.3) | 8.81 → 6.56 |
| 16K | 862 | 1178 | +36.7% (+36.6 – +36.8) | 19.0 → 13.9 |
| 32K | 789 | 1058 | +34.0% (+33.8 – +34.8) | 41.5 → 31.0 |

64K and 128K gave +31.9% (615 → 812 tok/s) and +23.7% (480 → 599 tok/s) over three pairs with a 15 s settle. Requests of 100–270 s heat the GPU during the request itself, which a settle period cannot remove, so these two rows are reference only.

Controls:

- A/A: two identical A8-off conditions at 4K, 8 tuples, differ by +0.11% on average (range −2.5% to +2.9%).
- Q8 A16: forcing the existing native Q8 A16 tile down to 2048-row chunks stays within about ±3% of stock MLX at 4K. The gain therefore comes from W8A8, not from lowering the 16384-row threshold of #3506.
- The gains sit near the bound implied by the routed GEMMs alone: microbenchmark times per call, multiplied by the per-chunk call counts, predict +33.7% if only those GEMMs changed.

Prefill chunks of 2048 never reach the Q8 A16 tile in the default configuration; on this model every Q8 projection ran on stock MLX before this change.

## Correctness

The centered GEMM is checked against an independent affine reference on random bytes with the edge codes 0, 127, 128 and 255, for FP16 and BF16 metadata, both activation scalings, single-row and partial tiles, and several tile variants. Decoded codes are bit-exact against an independent unpacker tied to MLX's dequantize.

Quality compares A16 and A8 activations on the same Q8 checkpoint: 36 sequences, 163,840 positions, chunk 2048. The corpus covers English, Korean, code, long prose, schema code, synthetic reasoning, tool-call transcripts, agent transcripts and JSON responses.

| Metric | Value |
|---|---:|
| ΔNLL, nats/token | −0.0021 |
| 95% CI over sequences | [−0.0062, +0.0019] |
| Perplexity ratio | 0.998 |
| Top-1 agreement | 0.945 |
| Mean KL | 0.222 |
| Greedy 24-token continuations identical | 22 / 36 |

The A16-versus-A16 control gives zero KL, and no NaN or Inf was observed. Aggregate NLL showed no measurable regression under this corpus. The distributional differences are concentrated in a small number of structured and tool-like positions:

- About 95% of the KL mass comes from the 2.4% of positions with KL > 1.
- Tool-call transcripts, agent transcripts and synthetic reasoning sequences show mean KL of 1.40, 0.75 and 0.047 with top-1 agreement of 0.82, 0.87 and 0.89. English, Korean, code, long prose and schema code stay at mean KL ≤ 0.002 with top-1 ≥ 0.97.
- ΔNLL in the tool and agent categories is near zero or negative, so the aggregate NLL does not show this shift. The per-category samples are small, so this describes the corpus, not a general bound.

A functional smoke test sent ten chat requests with tools (4.7K–10K prompt tokens, greedy), A8 off and on. Both conditions produced schema-valid tool calls in 10/10 cases, and the tool name and arguments matched in 9/10 (one title differs in letter case). Two requests chose a different tool than expected, identically in both conditions. Per-request counters confirm that A8 ran only in the enabled condition.

Q8A8 changes generated tokens in some cases. Similar aggregate NLL does not establish output equivalence or the absence of task-specific regressions.

## Memory

No unpacked weight copy is added. The packed weights are read in place. The persistent addition is the transposed GS64 scale and bias copies of eligible projections, as the existing Q4/Q5 A8 path also keeps.

| Measurement | A8 off, GiB | Q8A8, GiB | Difference |
|---|---:|---:|---:|
| Active, right after load | 27.942 | 27.942 | 0 |
| Active, settled after the first prefill | 27.942 | 29.358 | +1.416 |
| Peak, 16K prefill | 32.466 | 33.875 | +1.409 |
| Peak, 32K prefill | 33.443 | 34.904 | +1.461 |

This is a speed/memory tradeoff and is intentionally left as is. Reading the checkpoint's `[N, K/64]` metadata directly made the GEMM about 30% slower, and staging it through threadgroup memory made it 12–16% slower. Neither was adopted. Values are MLX active-memory counters, not process RSS.

## Validation

Candidate `5f3c536817fde75b1a0e6105c390a49d450d69c9` on upstream `25aebb3ee0bb052a264241953bef6820f102dd0b`. Performance, quality and memory measurements were taken on `1affe985` (based on `2238a444`); the candidate differs by the rebase and non-functional cleanup, and the Q8 kernel (apart from a license header) and the Q4/Q5 sources are identical. One tool-smoke case was re-run on the cleanup candidate.

| Check | Result |
|---|---|
| Targeted tests (`test_qwen35_oq_a8.py`, `test_qwen35_q4_mlp.py`, `test_m5_gather_qmm_a8.py`, native extension built) | 270 passed |
| CI command in a fresh Python 3.11 environment, no native extension | 17144 passed, 703 skipped, 0 failed; `main` in the same environment: 17144 passed, 669 skipped, 0 failed. The extra skips are native-kernel tests of this PR, which skip cleanly without the extension |
| Q4/Q5 A8 outputs on fixed inputs, main vs candidate build | 216 arrays bit-identical; Q4/Q5 kernel sources unchanged |
| `ruff check` on changed files | no new findings; the N8xx/SIM findings are the same on `main` |
| `black --check --diff` on changed files | no new findings; the remaining diffs in `fast.py` and `test_qwen35_q4_mlp.py` are identical on `main` |
| `git diff --check` | clean |

Tests cover Q8 GS64 classification, GS128 and Q6 rejection, the edge codes, FP16/BF16 metadata, the natural Stage A layout, the GEMM against the reference including partial M, tile agreement, no persistent weight copy, Stage A sharing in MLP and GDN, untiled-projection fallback, the standalone projection route and fallback, routing below the 16384-row floor, decode and short-row fallback, exclusion of the Q8 MTP head, and unchanged Q4/Q5 behavior. During MTP decode, 64 generated tokens issued no additional A8 calls, and a 4K prefill issued the same 800 A8 projection calls with MTP on as with it off.

## Reproduction

Evidence commit: https://github.com/hojin12312/omlx/commit/92cef5919b7a06f59138274a78406899b39a2eaa. Under `research/qwen38_q8a8/` it holds the harness, raw summaries, the throughput corpus and a report; the 36-sequence quality corpus (`sha256 2ee326fcd4ed3f2b92b5236a7c88c9eb48c9f3824766397de2ba64585d9c2faf`) is pinned by hash only.

```bash
OMLX_WITH_CUSTOM_KERNEL=1 python -m pip install -e .

# A8-off vs Q8A8 in one server, rotating order, long settle (4K/8K/16K shown)
# run from research/qwen38_q8a8/ of the evidence commit
python harness/guard_run.py --log run.log -- python -u harness/measure_q8.py \
  --python "$(command -v python)" --tree "$PWD" --workdir "$RUNS" \
  --model-dir "$MODELS" --checkpoint "$MODELS/Qwen3.8-27B-oQ8e-mtp" \
  --model Qwen3.8-27B-oQ8e-mtp --corpus harness/corpus.json \
  --conds off,on --rot 1 --lengths 4096 8192 16384 --samples 4 --settle 45 \
  --tag q8a8 --out "$RUNS/q8a8.json"
```

The driver uses its own isolated server configuration and in-process condition switching, outside the product code. `harness/run_quality.sh`, `run_memory.sh`, `run_mtp.sh` and `run_smoke.sh` reproduce the quality, memory, MTP and tool-smoke numbers.

## Checklist

- [ ] I read [CONTRIBUTING.md](https://github.com/jundot/omlx/blob/main/docs/CONTRIBUTING.md).
- [ ] I understand every change in this PR and can explain it in review.
- [x] No UI changes in this PR.

AI-assisted contribution.
