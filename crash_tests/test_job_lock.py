import subprocess
import sys
import threading

import pytest

from poster_vector_rebuilder.job_lock import job_lock


def test_nested_ownership_and_exception_release(tmp_path):
    with pytest.raises(ValueError):
        with job_lock(tmp_path):
            with job_lock(tmp_path / '.'):
                raise ValueError('injected stage failure')
    with job_lock(tmp_path):
        pass


def test_other_thread_rejected(tmp_path):
    errors = []
    def contender():
        try:
            with job_lock(tmp_path):
                errors.append('incorrectly acquired')
        except BlockingIOError:
            errors.append('busy')
    with job_lock(tmp_path):
        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors == ['busy']


def test_process_death_releases_ownership(tmp_path):
    code = 'import sys,time; from poster_vector_rebuilder.job_lock import job_lock;\nwith job_lock(sys.argv[1]):\n print("owned",flush=True)\n time.sleep(30)\n'
    child = subprocess.Popen([sys.executable, '-c', code, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'owned'
        with pytest.raises(BlockingIOError):
            with job_lock(tmp_path):
                pass
    finally:
        child.kill()
        child.communicate(timeout=5)
    with job_lock(tmp_path):
        pass
