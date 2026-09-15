from __future__ import annotations

import os
from pathlib import Path


class LockAlreadyHeld(RuntimeError):
    pass


class SingletonLock:
    """An OS-level advisory lock held for the lifetime of the backend process."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def acquire(self):
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                handle.write("0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            self._handle = handle
            return self
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise LockAlreadyHeld(f"Another backend already holds {self.path}") from exc

    def release(self):
        if self._handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
