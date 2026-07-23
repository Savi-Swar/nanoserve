"""Optional GPU utilization sampler. Polls nvidia-smi in a background thread
and reports mean/peak SM utilization over the run. No-ops (returns None) when
there's no NVIDIA GPU — e.g. on the Mac dev box — so the harness is portable
and only produces util numbers where they actually mean something.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time


class GpuSampler:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = shutil.which("nvidia-smi") is not None

    @property
    def available(self) -> bool:
        return self._available

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                self._samples.append(float(out.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        if self._available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()

    def summary(self) -> dict | None:
        if not self._available or not self._samples:
            return None
        s = self._samples
        return {"mean": sum(s) / len(s), "peak": max(s), "n": len(s)}
