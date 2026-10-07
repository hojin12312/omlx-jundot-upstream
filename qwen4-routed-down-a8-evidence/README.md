# Routed Down A8 evidence

Reproduction bundle, analysis summaries and validation receipt for the routed expert Down A8 prefill change
(Qwen3.8 Flash-Next, Apple M5). Nothing here is product code.

| Path | Content |
|---|---|
| `RECEIPT.md` | Pins (commits, hardware, toolchain, hashes), results and how they were produced |
| `analysis/` | Text summaries of the main campaign, the A/A control, the reference control, the instrumentation check, quality, memory and the test and static checks |
| `bundle/` | Matched-input driver, in-process hook, memory guard, corpus and aggregation scripts. `bundle/README.md` has the commands and `bundle/MANIFEST.json` the sha256 of every file |
| `raw/` | Per-request records of the main campaign and the controls (request and token-id hashes, `usage.prompt_tokens`, TTFT, counters). Local paths are replaced by `<WORK>`, `<MODEL_ROOT>`, `<HOME>` and `<TMP>`; nothing else is changed |

## Corpus

`bundle/corpus.json` holds 302,250 token ids built by `bundle/build_perf_corpus.py` from tracked files of the oMLX
repository (docs, READMEs and Python source under Apache-2.0). The file records the source commit, the git blob hash and token
count of every input file, the tokenizer hash and its own `sha256`. The model checkpoint is not included.

## Pinned commits

| Role | Commit |
|---|---|
| Measured reference | `4b6b4458a53dfb1a3c6ce2af7379f75ce13e24bd` |
| Measured candidate | `2d313a591fca71fe29a6746b70f0056d514e8a42` |
| Final branch base | `2238a44477a91123051d0e15ac8b3b4e8a95fa83` |
| Final branch head | `d392aa1c4dae27f50eb26d171217b1052c22f2b6` |

All timing, quality and memory numbers come from the measured reference and candidate. The final branch only adds the rebase
and the cleanup described in `RECEIPT.md`.

## Checking a number

Run these from the directory that holds the files. They print the tables behind `analysis/`.

```bash
cd raw/perf_main
python ../../bundle/analyze_matched.py paired paired_1.json paired_2.json

cd ../perf_controls
python ../../bundle/analyze_matched.py paired aa_1.json
python ../../bundle/analyze_matched.py control ctl_s1_ref.json ctl_s1_off.json ctl_s1_on.json ctl_s2_ref.json ctl_s2_off.json ctl_s2_on.json
python ../../bundle/analyze_matched.py control instr_s1_off.json instr_s1_on.json instr_s2_off.json instr_s2_on.json
```

The output equals `analysis/perf_main.txt`, `aa_control.txt`, `reference_control.txt` and `instrumentation.txt`, except that the
`control` headers list the files in the order given on the command line.

Quality, memory and the test logs are not re-derivable from `raw/`: `analysis/quality.txt` and `analysis/memory.txt` are summaries of
runs whose per-position arrays and logs are not included. The model checkpoint is not included either.
