# Down A8 matched-input reproduction bundle

Driver, hook, corpus and analysis for the prefill-throughput campaign of the routed Down A8 change
(`feat/qwen4-routed-down-a8`, Qwen3.8-Flash-Next, Apple M5). Nothing here is part of the product code.

| File | Purpose |
|---|---|
| `measure_prod.py` | matched-input driver (paired / A-A / single-condition modes) |
| `measure_pp2.py` | server lifecycle, tokenizer and the strict streamed-request timer it depends on |
| `prod_toggle/sitecustomize.py` | in-process hook: control FIFO with generation acks, in-memory counters |
| `smoke_guard_on.py` | normal-guard 8K serving smoke (scheduler memory guard on) |
| `smoke_tools.py` | tool-calling smoke, Down A16 vs A8 in one server (structural checks) |
| `guard_run.py` | memory guard: runs one model process and kills it before the machine thrashes |
| `build_perf_corpus.py` | rebuilds `corpus.json` from a checkout (provenance, per-file token counts) |
| `corpus.json` | 302,250 token ids of tracked Apache-2.0 oMLX files (docs, five READMEs, Python); `sha256` field inside |
| `analyze_matched.py` | aggregation (`paired` and `control` sub-commands) |
| `MANIFEST.json` | sha256 of every file above, the toolchain and the commit hashes of the runs |

The checkpoint is not included. The runs used `Jundot/Qwen3.8-Flash-Next-oQ4e-mtp`; `MODEL_ROOT`
must contain a directory named like `MODEL_ID` (a symlink is fine).

## Requirements and safety

* Apple M5-series Mac with 128 GiB and AC power; no other large model or GPU job. The process needs about
  100-107 GiB; run every model process through `guard_run.py`, one at a time. The guard refuses to start
  when memory is short, and it kills the process group when swap or the compressor grow or when
  decompression thrashes. It is a monitor, not a limit; do not copy the 112 GiB wired-limit setting of the
  measurement machine to a machine that was not validated for it.
* The driver starts an oMLX server with a dedicated `--base-path` under `--workdir`; the user's normal oMLX
  settings are not read or written. `OMLX_*` and `PYTHONPATH` of the calling shell are removed first.
* Scheduler `prefill_memory_guard` is **off** in the timing runs (as in all earlier rounds), so that the
  guard cannot change the chunking between conditions; the external monitor above guards the machine.

## Main campaign (the command of the PR body)

Set `BUNDLE`, `MODEL_ROOT` and the checkpoint name as in the PR body and run its script from a checkout of the
candidate. It performs two sessions of six matched pairs per length at 16384, 32768, 65536 and 131072 tokens
(the second session starts with the opposite condition); the raw receipts are `RUN_ROOT/paired_{1,2}.json`.

```bash
python "$BUNDLE/analyze_matched.py" paired "$RUN_ROOT/paired_1.json" "$RUN_ROOT/paired_2.json"
```

The script prints per-length throughput, the paired gain with its Student-t interval, per-run and per-order
gains, median TTFT, the hash/token-count matching and the counter coverage, and the table rows of the PR body.

## Control experiments

All use the same flags as the main campaign (`--python --tree --workdir --model-dir --checkpoint --model --corpus
corpus.json`) plus:

| Check | Flags |
|---|---|
| A/A (both labels Down A16) | `--mode aa --lengths 32768 131072 --samples 6 --tag aa_1` |
| Instrumentation off (candidate, per-process switch, no hook, 2 sessions, reversed order) | `--mode single --hook off --nonce-set instr --lengths 32768 --samples 6 --label off --down-env 0` and `--label on --down-env 1` |
| Reference control (reference tree, candidate Down off, candidate Down on; 2 sessions, reversed order) | `--mode single --nonce-set control --lengths 8192 32768 --samples 6` with `--tree REF --label ref`, `--tree CAND --label off --down-env 0`, `--tree CAND --label on --down-env 1` |

Aggregate single runs with `analyze_matched.py control RUN1.json RUN2.json ...`; prompts are matched across
processes by the sha256 of the request body.

## What a receipt contains

Per request: sha256 of the request body and of the re-tokenised prompt ids, `usage.prompt_tokens`, first-token
TTFT, `cached_tokens` (or its absence), power and thermal state, the generation acknowledged before the request
and the counter differences (A8 results per projection, enqueued launches by projection/G/`STAGED` mode,
`_stage_mode` queries, rows per Down call). Per run: argv, settings, environment, host, toolchain versions, the
sha256 of the A8 source files and native libraries actually imported, the corpus hash and every invalid sample
with its reason.
