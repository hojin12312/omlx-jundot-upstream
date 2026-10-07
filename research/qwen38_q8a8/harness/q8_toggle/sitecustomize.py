"""Research hook for the Q8A8 study (never part of the production tree).

Enable with ``OMLX_Q8_CTL=<dir>`` and ``PYTHONPATH=<tree>:<this dir>``. The server process then

  * serves a control FIFO ``<dir>/ctl.fifo`` from a daemon thread (blocked in open/read whenever
    no command is in flight, so it does nothing while a request is timed). One JSON command per
    line, acknowledged through ``<dir>/ack.json`` carrying the same ``gen``:

      {"op": "set",  "gen": n, "state": "off" | "on" | "a16"}
      {"op": "snap", "gen": n}     ack carries the in-memory counters and the file identities

    States (only ever switched between requests):
      on   the production wiring of the tree under test (oQ A8 wrappers and backends installed);
      off  the wrappers are removed again (MLP ``__call__`` restored, no GDN/linear backend), which
           is what a model loaded with ``qwen35_oq_a8_enabled`` off runs, minus the module tags;
      a16  like off, but the existing native Q8 A16 tile is forced to the same 2048-row floor that
           Q4/Q5 use (control for the 16384 threshold of issue #3506).

  * counts, in memory and identically in every state: Stage A calls, A8 projection calls by
    (bits, N, K), A8 MLP / GDN / standalone results and declines, native Q8 A16 calls, stock
    ``QuantizedLinear`` calls by (bits, N, K) and the row histogram of prefill-sized calls.
"""
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter

_CTL = os.environ.get("OMLX_Q8_CTL")
# Passive: count stock / native Q8 A16 calls and prefill rows only (default settings, no A8 patch).
_PASSIVE = os.environ.get("OMLX_Q8_PASSIVE") == "1"

if _CTL:
    STATE = {"state": None, "installed": False}
    COUNTS = Counter()
    PROJ = Counter()       # a8:q{bits}:N{n}:K{k}
    STOCK = Counter()      # stock:q{bits}:N{n}:K{k}   (rows >= 64 only)
    ROWS = Counter()       # stock + a8 prefill-sized rows, exact M
    META = {}
    SAVED = {}

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    def _install_stock(nn, fast):
        real_q8 = fast.qwen35_q8_affine_qmm_t

        def q8_native(x, weight, *a, **k):
            COUNTS["q8_a16_native"] += 1
            PROJ[f"a16native:q8:N{int(weight.shape[0])}:K{int(x.shape[-1])}"] += 1
            ROWS[int(x.shape[-2])] += 1
            return real_q8(x, weight, *a, **k)

        fast.qwen35_q8_affine_qmm_t = q8_native

        real_call = nn.QuantizedLinear.__call__

        def counted_call(self, x):
            rows = x.shape[-2] if x.ndim >= 2 else 1
            if rows >= 64:
                bits = int(getattr(self, "bits", 0))
                STOCK[f"stock:q{bits}:N{int(self.weight.shape[0])}:K{int(x.shape[-1])}"] += 1
                ROWS[int(rows)] += 1
                COUNTS["stock_proj"] += 1
            return real_call(self, x)

        nn.QuantizedLinear.__call__ = counted_call

    def _install():
        import mlx.nn as nn
        import omlx
        from omlx.custom_kernels.qwen35_prefill import fast
        from omlx.patches import qwen35_q4_mlp as q4

        if _PASSIVE:
            _install_stock(nn, fast)
            so = sorted({m.__file__ for m in list(sys.modules.values())
                         if getattr(m, "__file__", None) and m.__file__.endswith((".so", ".dylib"))
                         and "omlx" in m.__file__})
            META.update({"omlx_file": omlx.__file__, "oa_file": omlx.__file__, "pid": os.getpid(),
                         "native": {p: _sha256(p) for p in so}, "mlp_classes": 0, "passive": True,
                         "python": sys.version.split()[0]})
            STATE["installed"] = True
            return
        from omlx.patches import qwen35_oq_a8 as oa

        # ---- counting wrappers (same in every state) --------------------------------------
        real_stage = oa.stage_a

        def stage_a(x, *a, **k):
            COUNTS["stage_a"] += 1
            return real_stage(x, *a, **k)

        oa.stage_a = stage_a

        real_apply = oa.apply_plan

        def apply_plan(linear, stage, plan):
            out = real_apply(linear, stage, plan)
            COUNTS["a8_proj"] += 1
            n = int(linear.weight.shape[0])
            k = int(stage.qa.shape[-1])
            PROJ[f"a8:q{plan.bits}:N{n}:K{k}"] += 1
            ROWS[int(stage.qa.shape[-2])] += 1
            return out

        oa.apply_plan = apply_plan

        real_mlp = oa.oq_a8_mlp

        def oq_a8_mlp(*a, **k):
            out = real_mlp(*a, **k)
            COUNTS["mlp_a8" if out is not None else "mlp_decline"] += 1
            return out

        oa.oq_a8_mlp = oq_a8_mlp

        real_gdn = oa.oq_a8_gdn_projections

        def gdn_proj(*a, **k):
            out = real_gdn(*a, **k)
            COUNTS["gdn_a8" if out is not None else "gdn_decline"] += 1
            return out

        oa.oq_a8_gdn_projections = gdn_proj

        real_lin = oa._prefill_linear_backend

        def prefill_linear(linear, x):
            out = real_lin(linear, x)
            COUNTS["standalone_a8" if out is not None else "standalone_decline"] += 1
            return out

        oa._prefill_linear_backend = prefill_linear

        _install_stock(nn, fast)

        # ---- state switch -------------------------------------------------------------------
        mlp_classes = []
        for mod_name, cls_name in oa._MLP_TARGETS:
            mod = sys.modules.get(mod_name)
            cls = getattr(mod, cls_name, None) if mod else None
            if cls is not None and getattr(cls, "_omlx_oq_a8_patched", False):
                mlp_classes.append(cls)
        SAVED["mlp"] = [(c, c.__call__, c._omlx_oq_a8_original_call) for c in mlp_classes]
        SAVED["gdn"] = oa._gdn_prefill_backend
        SAVED["lin"] = oa._prefill_linear_backend
        SAVED["q4"] = (q4._VLMQuantizedPrefillLinear._route_q8_min_tokens, q4._Q8_BACKEND_MIN_ROWS
                       if hasattr(q4, "_Q8_BACKEND_MIN_ROWS") else None)
        SAVED["mods"] = (oa, q4, fast)
        STATE["installed"] = True
        so = sorted({m.__file__ for m in list(sys.modules.values())
                     if getattr(m, "__file__", None) and m.__file__.endswith((".so", ".dylib"))
                     and "omlx" in m.__file__})
        META.update({"omlx_file": omlx.__file__, "oa_file": oa.__file__, "pid": os.getpid(),
                     "native": {p: _sha256(p) for p in so}, "mlp_classes": len(mlp_classes),
                     "python": sys.version.split()[0]})

    def _apply_state(state):
        label = state
        state = "off" if state == "off2" else state  # off2: an A/A alias of off
        oa, q4, fast = SAVED["mods"]
        lm_backend = oa._gdn_prefill_backend if state == "on" else None
        lin_backend = oa._prefill_linear_backend if state == "on" else None
        q4.register_qwen35_lm_gdn_prefill_backend(lm_backend)
        q4.register_qwen35_prefill_linear_backend(lin_backend)
        for cls, patched, orig in SAVED["mlp"]:
            if state == "on":
                cls.__call__ = patched
            elif state == "off":
                cls.__call__ = orig
            else:  # a16: the q4 MLP wrapper with the Q8 floor lowered to the Q4/Q5 floor
                inner = getattr(cls, "_omlx_q4_mlp_original_call", None)
                cls.__call__ = (q4._make_patched_mlp(inner, 8, 2048, 2048) if inner is not None else orig)
        q4._VLMQuantizedPrefillLinear._route_q8_min_tokens = (
            2048 if state == "a16" else SAVED["q4"][0])
        STATE["state"] = label

    def _watch():
        deadline = time.time() + 3600
        while time.time() < deadline:
            mod = sys.modules.get("omlx.patches.qwen35_oq_a8")
            if _PASSIVE and "omlx.patches.qwen35_q4_mlp" in sys.modules and "omlx.engine.vlm" in sys.modules and (
                    "mlx_vlm.models.qwen3_5.language" in sys.modules):
                time.sleep(2.0)
                try:
                    _install()
                except Exception as exc:  # noqa: BLE001
                    META["install_error"] = repr(exc)
                return
            if (not _PASSIVE) and mod is not None and getattr(mod, "_MLP_PATCHED", False) and getattr(mod, "_GDN_REGISTERED", False):
                time.sleep(0.5)
                try:
                    _install()
                except Exception as exc:  # noqa: BLE001
                    META["install_error"] = repr(exc)
                return
            time.sleep(0.05)
        META["install_error"] = "the oQ A8 patch was never applied"

    def _write_ack(ack):
        path = os.path.join(_CTL, "ack.json")
        with open(path + ".tmp", "w") as f:
            json.dump(ack, f)
        os.replace(path + ".tmp", path)

    def _handle(cmd):
        ack = {"gen": cmd["gen"], "op": cmd["op"], "installed": STATE["installed"], "state": STATE["state"]}
        if cmd["op"] == "set" and STATE["installed"]:
            _apply_state(cmd["state"])
            ack["state"] = STATE["state"]
        elif cmd["op"] == "snap":
            ack.update({"counts": dict(COUNTS), "proj": dict(PROJ), "stock": dict(STOCK),
                        "rows": {str(k): v for k, v in ROWS.items()}, "meta": dict(META)})
        _write_ack(ack)

    def _serve():
        fifo = os.path.join(_CTL, "ctl.fifo")
        while True:
            try:
                with open(fifo) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            _handle(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                META["ctl_error"] = repr(exc)
                time.sleep(0.2)

    threading.Thread(target=_watch, daemon=True).start()
    threading.Thread(target=_serve, daemon=True).start()
