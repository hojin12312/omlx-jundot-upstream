"""Research harness hook for the production branch (never part of the production tree).

Enable with ``OMLX_PROD_CTL=<dir>`` and ``PYTHONPATH=<tree>:<this dir>``. The server process then

  * serves a control FIFO ``<dir>/ctl.fifo`` from a daemon thread that is blocked in ``open``/``read``
    whenever no command is in flight, so it does no work, and no file I/O, while a request is timed.
    The driver sends one JSON command per line and waits for ``<dir>/ack.json`` to carry the same
    ``gen``:

      {"op": "set",  "gen": n, "down": "0" | "1"}   switch routed Down A8 (only when
                                                     ``OMLX_PROD_TOGGLE=1``; "0" = A16 Down);
      {"op": "snap", "gen": n}                       ack carries the in-memory counters and the
                                                     identity of the imported files.

    Commands are only ever sent between requests. ``set`` replaces ``m5_gather_qmm_a8._down_enabled``
    by a function that returns an in-memory flag (the production switch is the
    ``OMLX_M5_ROUTED_DOWN_A8`` variable, which cannot change in a running server). Without
    ``OMLX_PROD_TOGGLE`` the production switch is untouched.

  * counts, in memory, with the same wrappers in every condition: the A8 calls that returned a result
    (Gate+Up and Down), declined Down calls, the rows of every Down call (the chunk-size evidence),
    kernel launches that were enqueued (by projection, G and ``STAGED`` mode) and the number of
    ``_stage_mode`` queries. A query is not a launch, and a launch is not an evaluated result: the
    work is evaluated when the request completes. The hot path only bumps integers.

The counters work on any tree that has ``omlx.patches.m5_gather_qmm_a8``; attributes a tree lacks
(for example the Down code of the reference tree) are skipped and simply count zero.
"""
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter

_CTL = os.environ.get("OMLX_PROD_CTL")
_TOGGLE = os.environ.get("OMLX_PROD_TOGGLE") == "1"

if _CTL:
    STATE = {"down": None, "installed": False}
    COUNTS = Counter()
    LAUNCH = Counter()
    MODE_QUERY = Counter()
    DOWN_ROWS = Counter()
    META = {}

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()

    def _install():
        import omlx
        from omlx.patches import m5_gather_qmm_a8 as a8

        real = a8.sorted_gather_qmm_a8
        real_mode = getattr(a8, "_stage_mode", None)
        real_launch = getattr(a8, "_launch", None)
        real_down = getattr(a8, "_down_enabled", None)

        def counting(*args, **kwargs):
            out = real(*args, **kwargs)
            if kwargs.get("swiglu"):
                if out is not None:
                    COUNTS["gate_up_a8_results"] += 1
            elif out is not None:
                COUNTS["down_a8_results"] += 1
                rows = int(args[4].shape[0])
                COUNTS["down_a8_rows"] += rows
                DOWN_ROWS[rows] += 1
            else:
                COUNTS["down_declined"] += 1
            return out

        a8.sorted_gather_qmm_a8 = counting

        if real_mode is not None:
            def counting_mode(groups, swiglu):
                mode = real_mode(groups, swiglu)
                MODE_QUERY[f"{'gate_up' if swiglu else 'down'}:G{groups}:mode{mode}"] += 1
                return mode

            a8._stage_mode = counting_mode

        if real_launch is not None:
            def counting_launch(x, *args, **kwargs):
                out = real_launch(x, *args, **kwargs)
                if out is not None:
                    try:
                        swiglu = bool(kwargs.get("swiglu"))
                        groups = int(x.shape[2]) // 64
                        mode = kwargs.get("mode")
                        if mode is None:
                            mode = real_mode(groups, swiglu) if real_mode is not None else "n/a"
                        LAUNCH[f"{'gate_up' if swiglu else 'down'}:G{groups}:mode{mode}"] += 1
                    except Exception:  # noqa: BLE001
                        LAUNCH["unclassified"] += 1
                return out

            a8._launch = counting_launch

        if _TOGGLE and real_down is not None:
            def toggled():
                v = STATE["down"]
                return real_down() if v is None else v

            a8._down_enabled = toggled

        so = sorted({m.__file__ for m in list(sys.modules.values())
                     if getattr(m, "__file__", None) and m.__file__.endswith((".so", ".dylib"))
                     and "omlx" in m.__file__})
        META.update({"omlx_file": omlx.__file__, "a8_file": a8.__file__, "pid": os.getpid(),
                     "native": {p: _sha256(p) for p in so},
                     "has_down_code": real_down is not None, "toggle": _TOGGLE,
                     "python": sys.version.split()[0]})
        STATE["installed"] = True

    def _watch():
        deadline = time.time() + 1800
        while time.time() < deadline:
            if "omlx.patches.m5_gather_qmm_a8" in sys.modules:
                time.sleep(0.5)
                try:
                    _install()
                except Exception as exc:  # noqa: BLE001
                    META["install_error"] = repr(exc)
                return
            time.sleep(0.05)
        META["install_error"] = "m5_gather_qmm_a8 was never imported"

    def _write_ack(ack):
        path = os.path.join(_CTL, "ack.json")
        with open(path + ".tmp", "w") as f:
            json.dump(ack, f)
        os.replace(path + ".tmp", path)

    def _handle(cmd):
        ack = {"gen": cmd["gen"], "op": cmd["op"], "installed": STATE["installed"],
               "down": STATE["down"]}
        if cmd["op"] == "set":
            STATE["down"] = {"1": True, "0": False}[str(cmd["down"])]
            ack["down"] = STATE["down"]
        elif cmd["op"] == "snap":
            ack.update({"counts": dict(COUNTS), "launch": dict(LAUNCH), "mode_query": dict(MODE_QUERY),
                        "down_rows": {str(k): v for k, v in DOWN_ROWS.items()}, "meta": dict(META)})
        _write_ack(ack)

    def _serve():
        fifo = os.path.join(_CTL, "ctl.fifo")
        while True:
            try:
                with open(fifo) as f:  # blocks until the driver opens it for writing
                    for line in f:
                        line = line.strip()
                        if line:
                            _handle(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                META["ctl_error"] = repr(exc)
                time.sleep(0.2)

    threading.Thread(target=_watch, daemon=True).start()
    threading.Thread(target=_serve, daemon=True).start()
