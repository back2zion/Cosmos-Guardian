"""Isolated browser-test server. Never used by the production server entry point."""
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread

from core.job_worker import Worker
from server import create_app

_workspace = tempfile.TemporaryDirectory(prefix='cosmos-browser-test-')


class BrowserInferenceStub:
    model = object()
    model_id = 'browser-test-stub'
    adapter_loaded = False
    use_label_grounding = False

    def analyze_media(self, media_path, safety_context):
        yield {'stage': 'complete', 'result': {
            'hazards_detected': [{'type': '끼임 위험', 'event_type': 'entrapment', 'severity': 'High',
                                  'description': '테스트용 정비 장면', 'reasoning': '테스트용 설명'}],
            'overall_safety_score': 30, 'recommendation': '방호장치 설치',
        }}


class BrowserBackend:
    def run(self, path, context, heartbeat):
        event = list(BrowserInferenceStub().analyze_media(path, context))[-1]
        return event['result'], {'model_id': 'browser-test-stub'}


@asynccontextmanager
async def lifespan(app):
    stopping = Event()
    worker = Worker(app.state.jobs, Path(_workspace.name) / 'uploads', BrowserBackend())
    def run():
        while not stopping.is_set():
            worker.run_once()
            stopping.wait(0.05)
    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopping.set()
        thread.join(timeout=5)


app = create_app(
    agent=BrowserInferenceStub(), db_path=Path(_workspace.name) / 'workflow.db',
    upload_dir=Path(_workspace.name) / 'uploads', inference_enabled=True, lifespan=lifespan,
    users={'a' * 32: {'id': 'author', 'role': 'operator'}, 'b' * 32: {'id': 'boss', 'role': 'supervisor'}},
)
