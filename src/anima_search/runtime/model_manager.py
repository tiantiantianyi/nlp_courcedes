from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from threading import RLock
from typing import Any


class ModelManager:
    def __init__(self, qwen_factory: Callable[[], Any], sd_factory: Callable[[], Any]) -> None:
        self.qwen_factory = qwen_factory
        self.sd_factory = sd_factory
        self.qwen = None
        self.sd = None
        self._lock = RLock()

    @staticmethod
    def _clear_cuda() -> None:
        try:
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @contextmanager
    def qwen_session(self):
        with self._lock:
            if self.sd is not None:
                if hasattr(self.sd, "unload"):
                    self.sd.unload()
                self.sd = None
                self._clear_cuda()
            if self.qwen is None:
                self.qwen = self.qwen_factory()
            yield self.qwen

    @contextmanager
    def sd_session(self):
        with self._lock:
            if self.qwen is not None:
                if hasattr(self.qwen, "unload"):
                    self.qwen.unload()
                self.qwen = None
                self._clear_cuda()
            if self.sd is None:
                self.sd = self.sd_factory()
            yield self.sd

    def unload_all(self) -> None:
        with self._lock:
            if self.qwen is not None and hasattr(self.qwen, "unload"):
                self.qwen.unload()
            if self.sd is not None and hasattr(self.sd, "unload"):
                self.sd.unload()
            self.qwen = self.sd = None
            self._clear_cuda()
