import hashlib
import json

import pytest

from core.jobs import JobStore
from core.workflow import WorkflowStore
from scripts.product_backup import backup, restore


def test_backup_restore_includes_pending_jobs_and_verified_media(tmp_path):
    store = WorkflowStore(tmp_path / 'live.db')
    uploads = tmp_path / 'uploads'
    uploads.mkdir()
    (uploads / 'frame.png').write_bytes(b'image')
    source = {'media_file': 'frame.png', 'input_sha256': hashlib.sha256(b'image').hexdigest()}
    jobs = JobStore(store)
    job, _ = jobs.submit(source, 'helmet area', {'id': 'author', 'role': 'operator'}, 'request-001')
    archive = tmp_path / 'backup'
    backup(store.path, uploads, archive)
    recovered = tmp_path / 'restored'
    restore(archive, recovered)
    assert JobStore(WorkflowStore(recovered / 'workflow.sqlite3')).get(job['id'])['state'] == 'queued'
    assert (recovered / 'uploads/frame.png').read_bytes() == b'image'
    with pytest.raises(ValueError, match='must not exist'):
        restore(archive, recovered)
    (archive / 'uploads/frame.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        restore(archive, tmp_path / 'corrupt-restore')
    assert not (tmp_path / 'corrupt-restore').exists()


def test_restore_rejects_path_traversal_before_writing(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    (archive / 'workflow.sqlite3').write_bytes(b'db')
    (archive / 'manifest.json').write_text(json.dumps({'format': 'cosmos-product-backup-v1', 'files': {
        'workflow.sqlite3': hashlib.sha256(b'db').hexdigest(), '../escape': 'a' * 64,
    }}))
    with pytest.raises(ValueError):
        restore(archive, tmp_path / 'restored')
    assert not (tmp_path / 'restored').exists()
