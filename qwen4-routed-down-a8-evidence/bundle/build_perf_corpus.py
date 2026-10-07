#!/usr/bin/env python3
"""Build the redistributable prefill-throughput corpus from files of an oMLX checkout (Apache-2.0).

The earlier perf runs cut their prompts from a quality corpus that contains private documents, which
cannot be shipped. This corpus uses only tracked files of the checkout under test: the English docs,
the five READMEs (English, Korean, Japanese, Chinese, French) and Python sources. It is deterministic:
files are sorted by path, interleaved one documentation file to three code files, every file is
truncated to a fixed token budget, and the stream stops at ``--target-tokens``.

Output (JSON): provenance (commit, blob ids, tokenizer hash, per-file token counts), the token ids,
and ``sha256`` of the canonical id list, which ``measure_prod.py`` recomputes and stores in every receipt.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True, help="oMLX checkout (a git worktree)")
ap.add_argument("--checkpoint", required=True, help="directory with tokenizer.json")
ap.add_argument("--target-tokens", type=int, default=300_000)
ap.add_argument("--doc-budget", type=int, default=8000)
ap.add_argument("--code-budget", type=int, default=5000)
ap.add_argument("--out", required=True)
args = ap.parse_args()

tree = Path(args.tree)


def git(*a):
    return subprocess.run(["git", "-C", str(tree), *a], capture_output=True, text=True, check=True).stdout


commit = git("rev-parse", "HEAD").strip()
blobs = {}
for line in git("ls-files", "-s").splitlines():
    meta, path = line.split("\t", 1)
    blobs[path] = meta.split()[1]
docs = sorted(p for p in blobs if (p.startswith("docs/") and p.endswith(".md")) or
              (p.startswith("README") and p.endswith(".md")))
code = sorted(p for p in blobs if p.startswith("omlx/") and p.endswith(".py"))
tok = Tokenizer.from_file(str(Path(args.checkpoint) / "tokenizer.json"))

order, di, ci = [], 0, 0
while di < len(docs) or ci < len(code):
    if di < len(docs):
        order.append((docs[di], args.doc_budget))
        di += 1
    for _ in range(3):
        if ci < len(code):
            order.append((code[ci], args.code_budget))
            ci += 1

tokens, files = [], []
for path, budget in order:
    text = (tree / path).read_text(errors="replace")
    ids = tok.encode(f"\n\n# file: {path}\n{text}").ids[:budget]
    tokens.extend(ids)
    files.append({"path": path, "git_blob": blobs[path], "tokens": len(ids)})
    if len(tokens) >= args.target_tokens:
        break

doc = {
    "kind": "oMLX prefill throughput corpus",
    "license": "Apache-2.0 (tracked files of the oMLX repository)",
    "source_commit": commit,
    "tokenizer_json_sha256": hashlib.sha256((Path(args.checkpoint) / "tokenizer.json").read_bytes()).hexdigest(),
    "construction": {"doc_budget": args.doc_budget, "code_budget": args.code_budget, "interleave": "1 doc : 3 code",
                     "target_tokens": args.target_tokens, "script": "build_perf_corpus.py"},
    "files": files,
    "token_count": len(tokens),
    "sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
    "tokens": tokens,
}
Path(args.out).write_text(json.dumps(doc))
print(f"{len(files)} files ({sum(f['path'].startswith(('docs', 'README')) for f in files)} docs), "
      f"{len(tokens)} tokens, sha256 {doc['sha256']}, commit {commit}")
