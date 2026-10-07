#!/usr/bin/env python3
"""Q8A8 memory in a fresh process per setting (the persistent metadata copies must not be shared by both arms).

``--mode off``: qwen35_oq_a8_enabled False (stock path, no wrappers). ``--mode on``: True (Q8A8).
Per process: engine load, active memory before any request, one warm-up prefill (builds the per-projection
metadata copies on first use), settled active after the warm-up cache is released, then for every length one
real chunked prefill (production chunk 2048) with peak / end-active / settled-active, read from the MLX allocator.
Persistent delta = settled(on) - settled(off) and the "before" values. Temporary A8 buffers show up as peak delta.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import mlx.core as mx

GiB = 1024 ** 3


def snap():
    return {"active": mx.get_active_memory() / GiB, "cache": mx.get_cache_memory() / GiB, "peak": mx.get_peak_memory() / GiB}


async def run(args):
    import omlx
    from omlx.custom_kernels.qwen35_prefill import _ext
    tree = args.tree.rstrip("/")
    assert omlx.__file__.startswith(tree + "/") and _ext.__file__.startswith(tree + "/"), omlx.__file__
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig

    ms = ModelSettings(qwen35_oq_a8_enabled=(args.mode == "on"), qwen35_oq_a8_min_tokens=128)
    mx.set_wired_limit(64 * GiB)
    eng = VLMBatchedEngine(model_name=args.ckpt, model_settings=ms, scheduler_config=SchedulerConfig())
    await eng.start()
    lm = eng._vlm_model.language_model
    base = json.load(open(args.corpus))["tokens"]
    out = {"mode": args.mode, "tree": tree, "chunk": args.chunk, "rows": [], "argv": __import__("sys").argv}
    mx.clear_cache()
    out["after_load"] = snap()

    def prefill(length, salt):
        off = (salt * 7919) % max(1, len(base) - length)
        toks = mx.array(base[off: off + length], dtype=mx.int32)[None, :]
        mx.clear_cache()
        pre = snap()
        mx.reset_peak_memory()
        cache = lm.make_cache()
        t0 = time.time()
        for s in range(0, length, args.chunk):
            o = lm(toks[:, s: s + args.chunk], cache=cache)
            mx.eval(o.logits)
            del o
        secs = time.time() - t0
        at_end = snap()
        del cache
        mx.clear_cache()
        return {"pre": pre, "at_end": at_end, "settled": snap(), "seconds": secs}

    out["warmup"] = prefill(4096, 1)
    out["settled_after_warmup"] = out["warmup"]["settled"]
    for i, length in enumerate(args.lengths):
        r = prefill(length, 10 + i)
        r["length"] = length
        out["rows"].append(r)
        print(f"{args.mode} L={length:>6} peak={r['at_end']['peak']:.3f} end={r['at_end']['active']:.3f} "
              f"settled={r['settled']['active']:.3f} {r['seconds']:.1f}s", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    Path(args.out).write_text(json.dumps(out, indent=1))
    try:
        await eng.stop()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--mode", required=True, choices=["off", "on"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", "-o", required=True)
    asyncio.run(run(ap.parse_args()))
