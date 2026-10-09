"""Run one durable queue worker per installation/GPU: python worker.py."""
import fcntl
import os
import signal
import time
from pathlib import Path

from core.job_worker import Worker
from core.jobs import JobStore
from core.workflow import WorkflowStore


def main():
    if os.getenv('COSMOS_INFERENCE_ENABLED', '1') != '1':
        raise SystemExit('Inference is disabled; no worker started')
    store = WorkflowStore(os.getenv('COSMOS_WORKFLOW_DB', 'data/workflow.sqlite3'))
    lock = Path(str(store.path) + '.worker.lock').open('a')  # noqa: SIM115 - closed after worker shutdown
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A worker already owns this installation') from None
    worker = Worker(JobStore(store), os.getenv('COSMOS_UPLOAD_DIR', 'uploads'))
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            if not worker.run_once():
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        worker.backend.close()
        lock.close()


if __name__ == '__main__':
    main()
