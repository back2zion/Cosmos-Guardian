import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from core.helmet import parse_classification
from core.job_worker import AnalysisFailure, InferenceProcess, Worker
from core.jobs import JobStore, QueueFull
from core.workflow import Conflict, WorkflowError, WorkflowStore
from server import create_app

USERS = {'a' * 32: {'id': 'author', 'role': 'operator'}, 'b' * 32: {'id': 'boss', 'role': 'supervisor'},
         'c' * 32: {'id': 'other', 'role': 'operator'}}
ACTOR = USERS['a' * 32]


@pytest.fixture
def setup(tmp_path):
    image = io.BytesIO()
    Image.new('RGB', (20, 20), 'white').save(image, format='PNG')
    app = create_app(db_path=tmp_path / 'workflow.db', upload_dir=tmp_path / 'uploads', users=USERS,
                     inference_enabled=True)
    with TestClient(app) as client:
        yield app, client, image.getvalue(), tmp_path


def submit(setup, key='request-001', content=None, actor='a', context='helmet area'):
    _, client, image, _ = setup
    return client.post('/jobs', files={'file': ('frame.png', image if content is None else content, 'image/png')},
                       data={'context': context}, headers={'Authorization': 'Bearer ' + actor * 32, 'Idempotency-Key': key})


def headers(actor='a'):
    return {'Authorization': 'Bearer ' + actor * 32}


class Backend:
    def run(self, path, context, heartbeat):
        assert heartbeat('inference')
        return parse_classification('YES'), {'model_id': 'test-only-stub', 'profile': 'helmet-v1'}


def test_submission_survives_restart_and_duplicate_submission_is_one_job(setup):
    app, client, _, root = setup
    first = submit(setup).json()
    second = submit(setup).json()
    assert first['id'] == second['id'] and second['reused_request']
    assert len(list((root / 'uploads').iterdir())) == 1
    assert second['duplicate_submissions'] == 1
    assert submit(setup, context='different').status_code == 409
    assert len(JobStore(WorkflowStore(root / 'workflow.db')).list(ACTOR)) == 1
    worker = Worker(app.state.jobs, root / 'uploads', Backend())
    assert worker.run_once() and not worker.run_once()
    job = client.get('/jobs/' + first['id'], headers=headers()).json()
    assert job['state'] == 'completed' and job['assessment_id'] == first['id']
    assert app.state.store.verify(job['assessment_id'])
    assert len(app.state.store.list()) == 1
    assert client.get('/ready').json()['worker_available']


def test_concurrent_claims_and_duplicate_submissions(setup):
    app, _, _, _ = setup
    source = {'input_sha256': 'abc', 'media_file': 'frame.png', 'input_name': 'frame.png'}
    jobs = app.state.jobs
    with ThreadPoolExecutor(max_workers=4) as pool:
        created = list(pool.map(lambda _: jobs.submit(source, 'scope', ACTOR, 'same-key')[0], range(4)))
        claims = list(pool.map(lambda _: jobs.claim(), range(4)))
    assert len({j['id'] for j in created}) == 1
    assert sum(j is not None for j in claims) == 1


def test_job_visibility_and_owner_only_controls(setup):
    _, client, _, _ = setup
    job = submit(setup).json()
    assert client.get('/jobs/' + job['id']).status_code == 401
    assert client.get('/jobs/' + job['id'], headers=headers('c')).status_code == 404
    assert client.get('/jobs', headers=headers('c')).json() == []
    assert client.get('/jobs/' + job['id'], headers=headers('b')).status_code == 200
    assert client.post('/jobs/' + job['id'] + '/cancel', headers=headers('b')).status_code == 403
    assert client.post('/jobs/' + job['id'] + '/cancel', headers=headers()).json()['state'] == 'cancelled'
    assert client.post('/jobs/' + job['id'] + '/retry', headers=headers()).json()['state'] == 'queued'


def test_cancelled_and_expired_claims_cannot_create_assessments(setup):
    app, _, _, _ = setup
    jobs = app.state.jobs
    submitted = submit(setup).json()
    old = jobs.claim()
    jobs.control(submitted['id'], ACTOR, 'cancel')
    jobs.control(submitted['id'], ACTOR, 'retry')
    new = jobs.claim()
    with pytest.raises(Conflict):
        jobs.finish(old['id'], old['claim'], parse_classification('YES'), {}, 1)
    assert app.state.store.list() == []
    jobs.finish(new['id'], new['claim'], parse_classification('UNKNOWN'), {}, 1)
    document = app.state.store.get(new['id'])
    assert document['original_result']['assessment_status'] == 'unknown'
    assert 'overall_safety_score' not in document['original_result']


def test_worker_crash_is_explicit_failure_and_requires_retry(setup):
    app, _, _, _ = setup
    jobs = app.state.jobs
    submit(setup)
    old = jobs.claim()
    with jobs.workflow.connect() as db:
        job = jobs._load(db, old['id'])
        job['lease_until'] = 0
        jobs._save(db, job)
    assert jobs.claim() is None
    assert jobs.get(old['id'])['error'] == 'worker_lost'
    with pytest.raises(Conflict):
        jobs.finish(old['id'], old['claim'], parse_classification('NO'), {}, 1)


def test_retry_limit_prevents_unbounded_inference_and_preserves_failures(setup):
    app, _, _, _ = setup
    jobs = app.state.jobs
    submit(setup)
    for attempt in range(1, 4):
        job = jobs.claim()
        assert job['attempt'] == attempt
        jobs.fail(job['id'], job['claim'], 'timeout', 5)
        if attempt < 3:
            jobs.control(job['id'], ACTOR, 'retry')
    with pytest.raises(Conflict, match='3회'):
        jobs.control(job['id'], ACTOR, 'retry')
    final = jobs.get(job['id'])
    assert len(final['attempts']) == 3
    assert sum(item['elapsed_seconds'] for item in final['attempts']) == 15
    assert app.state.store.list() == []


def test_invalid_result_rolls_back_job_and_assessment_together(setup):
    app, _, _, _ = setup
    submit(setup)
    job = app.state.jobs.claim()
    with pytest.raises(WorkflowError):
        app.state.jobs.finish(job['id'], job['claim'], {'error': 'bad'}, {}, 1)
    assert not app.state.store.list()
    assert app.state.jobs.get(job['id'])['state'] == 'running'


def test_changed_upload_never_reaches_model(setup):
    app, _, _, root = setup
    job = submit(setup).json()
    (root / 'uploads' / job['source']['media_file']).write_bytes(b'changed')
    worker = Worker(app.state.jobs, root / 'uploads', Backend())
    assert worker.run_once()
    assert app.state.jobs.get(job['id'])['error'] == 'source_changed_or_missing'
    assert not app.state.store.list()


def test_invalid_images_and_overloaded_queue_leave_no_orphans(setup):
    app, _, _, root = setup
    assert submit(setup, content=b'not an image').status_code == 422
    assert not list((root / 'uploads').iterdir())
    app.state.jobs.max_pending = 1
    submit(setup)
    assert submit(setup, key='new-key-002').status_code == 429
    assert len(list((root / 'uploads').iterdir())) == 1
    with pytest.raises(QueueFull):
        app.state.jobs.submit({'input_sha256': hashlib.sha256(b'new').hexdigest()}, '', ACTOR, 'key-3')


def test_manual_draft_works_without_inference_and_never_enters_gpu_queue(setup):
    _, _, image, root = setup
    app = create_app(db_path=root / 'manual.db', upload_dir=root / 'manual-uploads', users=USERS,
                     inference_enabled=False)
    with TestClient(app) as client:
        request_headers = {**headers(), 'Idempotency-Key': 'manual-request-001'}
        def send(mode):
            return client.post('/jobs', headers=request_headers, data={'mode': mode},
                               files={'file': ('frame.png', image, 'image/png')})
        assert send('helmet').status_code == 503
        first = send('manual')
        assert first.status_code == 202
        job = first.json()
        assert job['state'] == 'completed' and job['attempt'] == 0 and job['attempts'] == []
        assert send('manual').json()['id'] == job['id']
        assert app.state.jobs.claim() is None
        document = app.state.store.get(job['assessment_id'])
        assert document['original_result']['analysis_performed'] is False
        assert document['original_result']['assessment_status'] == 'unknown'
        assert document['source']['model_id'] is None
        assert app.state.store.verify(document['id'])
        assert len(list((root / 'manual-uploads').iterdir())) == 1
        assert client.get('/ready').json()['ready']


def test_manual_mode_is_in_idempotency_contract_and_bypasses_gpu_capacity(setup):
    app, _, _, _ = setup
    jobs = app.state.jobs
    jobs.max_pending = 0
    source = {'input_sha256': 'abc', 'media_file': 'frame.png'}
    manual, _ = jobs.submit(source, '', ACTOR, 'shared-request', profile='manual-v1')
    with pytest.raises(Conflict):
        jobs.submit(source, '', ACTOR, 'shared-request', profile='helmet-v1')
    assert jobs.get(manual['id'])['state'] == 'completed'


def hanging_child(connection):
    connection.recv()
    while True:
        time.sleep(0.05)


def test_timeout_kills_inference_process_without_poisoning_api():
    backend = InferenceProcess(timeout_seconds=0.25, target=hanging_child)
    with pytest.raises(AnalysisFailure, match='timeout'):
        backend.run('unused', '', lambda _: True)
    assert backend.process is None


def test_cancel_terminates_inference_process():
    backend = InferenceProcess(timeout_seconds=30, target=hanging_child)
    with pytest.raises(AnalysisFailure, match='cancelled'):
        backend.run('unused', '', lambda _: False)
    assert backend.process is None
