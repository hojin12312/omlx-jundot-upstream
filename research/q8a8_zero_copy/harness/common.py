"""Shared helpers copied from the earlier Q8A8 harness (measure_q8.py): sh() and the control-FIFO driver."""
import json, os, subprocess, time
from pathlib import Path


def sh(*cmd):
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


class Ctl:
    """Driver side of the control FIFO: one command, then wait for the matching acknowledgement."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.fifo = directory / "ctl.fifo"
        self.ack = directory / "ack.json"
        self.gen = 0
        directory.mkdir(parents=True, exist_ok=True)
        os.mkfifo(self.fifo)

    def call(self, op, timeout=60.0, **kw):
        self.gen += 1
        line = (json.dumps({"op": op, "gen": self.gen, **kw}) + "\n").encode()
        deadline = time.time() + timeout
        while True:
            try:
                fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)  # ENXIO until the hook listens
                break
            except OSError:
                if time.time() > deadline:
                    raise SystemExit("control FIFO has no reader: the hook is not loaded")
                time.sleep(0.05)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        while time.time() < deadline:
            try:
                ack = json.loads(self.ack.read_text())
                if ack.get("gen") == self.gen:
                    return ack
            except (OSError, ValueError):
                pass
            time.sleep(0.005)
        raise SystemExit(f"no acknowledgement for generation {self.gen} ({op})")
