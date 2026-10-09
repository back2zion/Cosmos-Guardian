"""A persistent model child, killed on cancellation/timeout and restarted on demand."""
import multiprocessing
import os
import time
from pathlib import Path
from uuid import uuid4

from .audit import sha256_file
from .workflow import Conflict


class AnalysisFailure(RuntimeError):
    pass


def inference_child(connection):
    # CUDA is imported only inside a spawn child, never forked from an API process.
    os.environ.setdefault('HF_HUB_CACHE', str(Path(__file__).resolve().parents[1] / 'data/huggingface'))
    from .helmet_agent import HelmetAgent
    agent = HelmetAgent(use_label_grounding=False)
    try:
        while True:
            path, context = connection.recv()
            try:
                result = None
                for event in agent.analyze_media(path, safety_context=context):
                    if event['stage'] == 'complete':
                        result = event['result']
                    else:
                        connection.send({'stage': event['stage']})
                if result is None or result.get('error'):
                    connection.send({'error': 'invalid_model_output'})
                    continue
                connection.send({'result': result, 'metadata': {
                    'model_id': agent.model_id, 'model_revision': getattr(agent.model.config, '_commit_hash', None),
                    'adapter_loaded': False, 'label_grounding': False, 'profile': agent.profile,
                    'prompt_sha256': sha256_file(Path(__file__).resolve().parents[1] / 'prompts/helmet_inspector.yaml'),
                }})
            except Exception:  # noqa: BLE001 - contain inference errors without returning private tracebacks
                connection.send({'error': 'inference_failed'})
                # A failed CUDA context is not reused for another job.
                break
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class InferenceProcess:
    def __init__(self, timeout_seconds=300, target=inference_child):
        self.timeout = timeout_seconds
        self.target = target
        self.process = None
        self.connection = None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def run(self, path, context, heartbeat):
        if self.process is None or not self.process.is_alive():
            self.close()
            runtime = multiprocessing.get_context('spawn')
            self.connection, child = runtime.Pipe()
            self.process = runtime.Process(target=self.target, args=(child,), daemon=True)
            self.process.start()
            child.close()
        started = time.monotonic()
        try:
            self.connection.send((str(path), context))
            while True:
                if not heartbeat(None):
                    raise AnalysisFailure('cancelled')
                if time.monotonic() - started > self.timeout:
                    raise AnalysisFailure('timeout')
                if self.connection.poll(0.2):
                    event = self.connection.recv()
                    if 'error' in event:
                        raise AnalysisFailure(event['error'])
                    if 'result' in event:
                        return event['result'], event['metadata']
                    if not heartbeat(event.get('stage')):
                        raise AnalysisFailure('cancelled')
                elif not self.process.is_alive():
                    raise AnalysisFailure('worker_process_exited')
        except (EOFError, BrokenPipeError, OSError) as exc:
            self.close()
            raise AnalysisFailure('worker_process_exited') from exc
        except BaseException:
            self.close()
            raise


class Worker:
    def __init__(self, jobs, uploads, backend=None):
        self.jobs = jobs
        self.uploads = Path(uploads)
        self.backend = backend or InferenceProcess(int(os.getenv('COSMOS_JOB_TIMEOUT_SECONDS', '300')))
        self.id = uuid4().hex

    def run_once(self):
        self.jobs.pulse(self.id)
        job = self.jobs.claim()
        if job is None:
            return False
        started = time.monotonic()
        last_pulse = 0

        def heartbeat(stage):
            nonlocal last_pulse
            if stage or time.monotonic() - last_pulse > 2:
                self.jobs.pulse(self.id)
                last_pulse = time.monotonic()
                return self.jobs.heartbeat(job['id'], job['claim'], stage)
            current = self.jobs.get(job['id'])
            return current['state'] == 'running' and current.get('claim') == job['claim']

        try:
            filename = job['source']['media_file']
            if Path(filename).name != filename:
                raise AnalysisFailure('source_invalid')
            path = self.uploads / filename
            if not path.is_file() or sha256_file(path) != job['source']['input_sha256']:
                raise AnalysisFailure('source_changed_or_missing')
            result, metadata = self.backend.run(path, job['context'], heartbeat)
            self.jobs.finish(job['id'], job['claim'], result, metadata, time.monotonic() - started)
        except Conflict:
            self.jobs.fail(job['id'], job['claim'], 'cancelled', time.monotonic() - started)
        except AnalysisFailure as exc:
            self.jobs.fail(job['id'], job['claim'], str(exc), time.monotonic() - started)
        except Exception:  # noqa: BLE001 - persist failures without inventing successful assessments
            self.jobs.fail(job['id'], job['claim'], 'processing_failed', time.monotonic() - started)
        return True
