"""Research hook for the measured Q8A8 memory study (never part of the production tree).

With ``OMLX_DEC_CTL=<dir>`` the server process serves a control FIFO ``<dir>/ctl.fifo``
(one JSON command per line, acknowledged through ``<dir>/ack.json`` with the same ``gen``):

  {"op": "mem", "gen": n, "reset_peak": bool}  MLX active/cache/peak plus *live* phys_footprint
  {"op": "flags", "gen": n}                    patch flags that show Q8A8 is wired, pid, module files
  {"op": "install", "gen": n}                  wrap the A8 route + clear_cache with counters (idempotent)
  {"op": "counters", "gen": n, "reset": bool}  counters since install/reset, plus the clear_cache event log
  {"op": "lifetime", "gen": n}                 stats of the metadata-lifetime layer, if the tree has one

phys_footprint is read with proc_pid_rusage at the moment of the call (not from a cached sample).
The wrappers add one dict increment per A8 projection call and are installed identically in every arm.
"""
import ctypes
import ctypes.util
import json
import os
import sys
import threading
import time

_CTL = os.environ.get("OMLX_DEC_CTL")

if _CTL:
    _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    _COUNTS: dict = {}
    _CLEARS: list = []
    _ROWS: dict = {}  # Q8 A8 rows (M) per call -> number of calls
    _INSTALLED = {"v": False}
    _T0 = time.perf_counter()

    def _footprint() -> int:
        buf = ctypes.create_string_buffer(512)  # rusage_info_v4 is 464 bytes
        rc = _libproc.proc_pid_rusage(os.getpid(), 4, buf)
        if rc != 0:
            return -1
        return int.from_bytes(buf.raw[72:80], "little")  # ri_phys_footprint

    def _flags():
        oa = sys.modules.get("omlx.patches.qwen35_oq_a8")
        fast = sys.modules.get("omlx.custom_kernels.qwen35_prefill.fast")
        return {"pid": os.getpid(), "oq_a8_module_loaded": oa is not None,
                "mlp_patched": bool(getattr(oa, "_MLP_PATCHED", False)),
                "gdn_registered": bool(getattr(oa, "_GDN_REGISTERED", False)),
                "omlx_file": getattr(sys.modules.get("omlx"), "__file__", None),
                "ext_file": getattr(getattr(fast, "_ext", None), "__file__", None),
                "oq_a8_available": bool(fast.oq_a8_available()) if fast else None}

    def _bump(key, n=1):
        _COUNTS[key] = _COUNTS.get(key, 0) + n

    def _install():
        if _INSTALLED["v"]:
            return True
        import mlx.core as mx
        oa = sys.modules.get("omlx.patches.qwen35_oq_a8")
        fast = sys.modules.get("omlx.custom_kernels.qwen35_prefill.fast")
        if oa is None or fast is None:
            return False
        attr = getattr(oa, "_PREPARED_ATTR", "_omlx_oq_a8_prepared")
        orig_prep = oa._prepared_weights
        orig_qmm = fast.qwen35_oq_a8_qmm_t
        orig_clear = mx.clear_cache

        def prep(linear, *a, **k):
            # apply_plan passes native_meta positionally in the follow-up tree and by keyword in the first
            if k.get("native_meta") or (a and a[0] is True):
                # zero-copy route: the layer's own arrays, nothing may be built or cached
                r = orig_prep(linear, *a, **k)
                _bump("prepare_native")
                if getattr(linear, attr, None) is not None:
                    _bump("prepare_native_cached_copy")  # must stay 0
                return r
            miss = getattr(linear, attr, None) is None
            r = orig_prep(linear, *a, **k)
            _bump("prepare_calls")
            if miss and not hasattr(linear, "packed_weight"):
                _bump("prepare_builds")
            return r

        def qmm(*a, **k):
            _bump("a8_qmm_calls")
            bits = a[6] if len(a) > 6 else k.get("bits")
            if bits == 8:
                nat = bool(k.get("native_meta"))
                _bump("a8_q8_native" if nat else "a8_q8_transposed")
                kdim = a[3].shape[1] * 4  # weight is uint32 [N, K/4]
                rows = int(a[0].size // kdim)
                _ROWS[rows] = _ROWS.get(rows, 0) + 1
            else:
                _bump("a8_other_bits")
            return orig_qmm(*a, **k)

        def clear():
            before = mx.get_cache_memory()
            t = time.perf_counter()
            r = orig_clear()
            _CLEARS.append({"t": t - _T0, "dt_ms": (time.perf_counter() - t) * 1e3,
                            "cache_before": int(before), "cache_after": int(mx.get_cache_memory())})
            _bump("clear_cache_calls")
            return r

        oa._prepared_weights = prep
        fast.qwen35_oq_a8_qmm_t = qmm
        mx.clear_cache = clear
        _INSTALLED["v"] = True
        return True

    def _handle(cmd):
        ack = {"gen": cmd["gen"], "op": cmd["op"]}
        op = cmd["op"]
        if op == "mem":
            import mlx.core as mx
            if cmd.get("reset_peak"):
                mx.reset_peak_memory()
            ack.update({"active": int(mx.get_active_memory()), "cache": int(mx.get_cache_memory()),
                        "peak": int(mx.get_peak_memory()), "phys_footprint": _footprint(),
                        "t": time.perf_counter() - _T0})
        elif op == "flags":
            ack.update(_flags())
        elif op == "install":
            ack["installed"] = _install()
        elif op == "counters":
            ack["counts"] = dict(_COUNTS)
            ack["clear_events"] = list(_CLEARS)
            ack["rows_hist"] = {str(k): v for k, v in sorted(_ROWS.items())}
            if cmd.get("reset"):
                _COUNTS.clear()
                _CLEARS.clear()
                _ROWS.clear()
        elif op == "set_native":
            # study knob: switch every Q8 projection of this process between the two kernels (same weights, server, build).
            # native=True also drops the group-major copies so the arm holds the same memory as a fresh server.
            # Works on the intermediate tree (env switch) and on trees without the switch (plans are edited in place).
            import dataclasses
            import gc
            oa = sys.modules.get("omlx.patches.qwen35_oq_a8")
            flag = bool(cmd["native"])
            plan_attr = oa._PLAN_ATTR
            prep_attr = oa._PREPARED_ATTR
            n_plans = dropped = 0
            for o in gc.get_objects():
                try:
                    dct = o.__dict__ if hasattr(o, "__dict__") and not isinstance(o, type) else None
                except Exception:  # noqa: BLE001
                    continue
                if not dct:
                    continue
                plan = dct.get(plan_attr)
                if plan is not None and plan.bits == 8 and not plan.packed and plan.variant == 806:
                    dct[plan_attr] = dataclasses.replace(plan, native_meta=flag)
                    n_plans += 1
                if flag and prep_attr in dct and not hasattr(o, "packed_weight"):
                    del dct[prep_attr]
                    dropped += 1
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()
            ack.update({"replanned": n_plans, "dropped_prepared": dropped, "native": flag})
        elif op == "set_plans":
            # study knob: switch the Q8 projections of one family between the two kernels without touching the rest.
            # subset: all | mlp (gate/up/down, 17408 wide) | rest. Projections outside the subset stay group-major.
            import dataclasses
            import gc
            oa = sys.modules.get("omlx.patches.qwen35_oq_a8")
            flag = bool(cmd["native"])
            subset = cmd.get("subset", "all")
            plan_attr = oa._PLAN_ATTR
            n_native = n_trans = 0
            for o in gc.get_objects():
                try:
                    dct = o.__dict__ if hasattr(o, "__dict__") and not isinstance(o, type) else None
                except Exception:  # noqa: BLE001
                    continue
                if not dct or dct.get(plan_attr) is None:
                    continue
                plan = dct[plan_attr]
                if plan.bits != 8 or plan.packed or plan.variant != 806:
                    continue
                w = getattr(o, "weight", None)
                if w is None:
                    continue
                n_out, k_in = w.shape[0], w.shape[1] * 4
                is_mlp = 17408 in (n_out, k_in)
                member = subset == "all" or (subset == "mlp" and is_mlp) or (subset == "rest" and not is_mlp)
                want = flag and member
                dct[plan_attr] = dataclasses.replace(plan, native_meta=want)
                n_native += want
                n_trans += not want
            ack.update({"native_modules": n_native, "transposed_modules": n_trans, "subset": subset, "native": flag})
        elif op == "set_chunk":
            # study knob: the server exposes no prefill chunk size setting, so the scheduler's base size is wrapped
            sched = sys.modules.get("omlx.scheduler")
            size = int(cmd["size"])
            if not hasattr(sched.Scheduler, "_study_orig_base"):
                sched.Scheduler._study_orig_base = sched.Scheduler._base_prefill_step_size
            sched.Scheduler._base_prefill_step_size = (
                (lambda self, processed, remaining: size) if size else sched.Scheduler._study_orig_base)
            ack["chunk"] = size
        elif op == "scan_prepared":
            import gc
            oa = sys.modules.get("omlx.patches.qwen35_oq_a8")
            attr = getattr(oa, "_PREPARED_ATTR", "_omlx_oq_a8_prepared")
            n = nbytes = 0
            for o in gc.get_objects():
                try:
                    pr = o.__dict__.get(attr) if hasattr(o, "__dict__") and not isinstance(o, type) else None
                except Exception:  # noqa: BLE001
                    continue
                if pr is not None:
                    n += 1
                    nbytes += int(pr[1].nbytes + pr[2].nbytes)
            ack["layers_with_prepared_copy"] = n
            ack["prepared_copy_bytes"] = nbytes
        elif op == "lifetime":
            oa = sys.modules.get("omlx.patches.qwen35_oq_a8")
            fn = getattr(oa, "lifetime_stats", None)
            ack["stats"] = fn() if fn else None
        path = os.path.join(_CTL, "ack.json")
        with open(path + ".tmp", "w") as f:
            json.dump(ack, f)
        os.replace(path + ".tmp", path)

    def _serve():
        fifo = os.path.join(_CTL, "ctl.fifo")
        while True:
            try:
                with open(fifo) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                _handle(json.loads(line))
                            except Exception as exc:  # noqa: BLE001
                                path = os.path.join(_CTL, "ack.json")
                                with open(path + ".tmp", "w") as g:
                                    json.dump({"gen": json.loads(line).get("gen"), "error": repr(exc)}, g)
                                os.replace(path + ".tmp", path)
            except Exception:  # noqa: BLE001
                time.sleep(0.2)

    threading.Thread(target=_serve, daemon=True).start()
