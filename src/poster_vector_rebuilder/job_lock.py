"""Nonblocking, reentrant ownership of a job's fixed output filenames."""
from contextlib import contextmanager
from functools import wraps
import inspect
import os
from pathlib import Path
import threading

_held = threading.local()


@contextmanager
def job_lock(directory):
    directory = Path(directory).resolve()
    key = (os.getpid(), str(directory))
    held = getattr(_held, 'directories', None)
    if held is None:
        held = _held.directories = set()
    if key in held:
        yield
        return
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the inode: unlinking can split owners between old and new inodes.
    # The OS releases ownership even after process death.
    with (directory / '.poster-vector.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            acquire = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            acquire = lambda: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(handle, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise BlockingIOError(f'Job directory is already in use: {directory}') from exc
        held.add(key)
        try:
            yield
        finally:
            held.remove(key)
            release()


def exclusive_job(parameter):
    """Protect the full operation, including nested stages on the same thread."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            with job_lock(bound.arguments[parameter]):
                return function(*args, **kwargs)
        return wrapped
    return decorate
