"""Exercise the built local containers against one real image in isolated disposable volumes.

No mock predictions, supervisor decisions, or production records are created.
Requires already-built cosmos-guardian-{api,worker}:dev images and Docker GPU support.
"""
import argparse
import hashlib
import json
import secrets
import subprocess
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import httpx


def docker(*args):
    completed = subprocess.run(['docker', *args], check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--cache', type=Path, default=Path('data/huggingface'))
    parser.add_argument('--output', type=Path, default=Path('outputs/product-smoke.json'))
    args = parser.parse_args()
    content = args.image.read_bytes()
    name = 'cosmos-smoke-' + uuid4().hex[:12]
    api_name, worker_name = name + '-api', name + '-worker'
    volumes = [name + '-data', name + '-uploads']
    token = secrets.token_urlsafe(32)
    containers = []
    created_volumes = []
    result = {'status': 'failed', 'input_sha256': hashlib.sha256(content).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.NamedTemporaryFile(mode='w', prefix='cosmos-smoke-', suffix='.env') as env:
            env.write('COSMOS_USERS=' + json.dumps({token: {'id': 'smoke-operator', 'role': 'operator'}}) + '\n')
            env.write('COSMOS_WORKFLOW_DB=/app/data/workflow.sqlite3\nCOSMOS_UPLOAD_DIR=/app/uploads\n')
            env.flush()
            for volume in volumes:
                docker('volume', 'create', volume)
                created_volumes.append(volume)
            common = ['--env-file', env.name, '--cap-drop=ALL', '--security-opt=no-new-privileges',
                      '-v', volumes[0] + ':/app/data', '-v', volumes[1] + ':/app/uploads']
            docker('run', '-d', '--name', api_name, '-p', '127.0.0.1::8888', *common, 'cosmos-guardian-api:dev')
            containers.append(api_name)
            endpoint = 'http://' + docker('port', api_name, '8888/tcp') + '/api'
            with httpx.Client(timeout=20) as client:
                for _ in range(60):
                    try:
                        if client.get(endpoint + '/health').status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError('API did not start')
                assert client.get(endpoint.removesuffix('/api') + '/').status_code == 200
                assert client.get(endpoint + '/jobs').status_code == 401
                assert client.get(endpoint + '/ready').status_code == 503
                headers = {'Authorization': 'Bearer ' + token, 'Idempotency-Key': name}
                manual = client.post(endpoint + '/jobs',
                                     headers={**headers, 'Idempotency-Key': name + '-manual'},
                                     data={'mode': 'manual'}, files={'file': (args.image.name, content, 'image/png')})
                manual.raise_for_status()
                assert manual.json()['state'] == 'completed' and manual.json()['attempt'] == 0
                result['manual_without_worker'] = manual.json()
                def submit():
                    response = client.post(endpoint + '/jobs', headers=headers,
                                           files={'file': (args.image.name, content, 'image/png')})
                    response.raise_for_status()
                    return response.json()
                job = submit()
                duplicate = submit()
                assert duplicate['id'] == job['id'] and duplicate['reused_request']
                docker('restart', api_name)
                # The worker and API share SQLite and immutable source files, not Python memory.
                docker('run', '-d', '--name', worker_name, '--gpus', 'all', *common,
                       '-v', str(args.cache.resolve()) + ':/app/data/huggingface:ro',
                       'cosmos-guardian-worker:dev')
                containers.append(worker_name)
                deadline = time.monotonic() + 360
                while time.monotonic() < deadline:
                    try:
                        response = client.get(endpoint + '/jobs/' + job['id'], headers=headers)
                        response.raise_for_status()
                        job = response.json()
                        if job['state'] in {'completed', 'failed', 'cancelled'}:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(2)
                result['job'] = job
                assert job['state'] == 'completed', 'Real inference did not complete'
                assessment = client.get(endpoint + '/assessments/' + job['assessment_id'], headers=headers)
                assessment.raise_for_status()
                result['assessment'] = assessment.json()
                media = client.get(endpoint + '/assessments/' + job['assessment_id'] + '/media', headers=headers)
                media.raise_for_status()
                assert hashlib.sha256(media.content).hexdigest() == result['input_sha256']
                integrity = client.get(endpoint + '/assessments/' + job['assessment_id'] + '/integrity', headers=headers)
                integrity.raise_for_status()
                result['integrity'] = integrity.json()
                assert result['integrity']['valid'] is True
                assert client.get(endpoint + '/ready').status_code == 200
                assert len(client.get(endpoint + '/jobs', headers=headers).json()) == 2
                docker('exec', api_name, 'python', 'scripts/product_backup.py', 'backup',
                       '--database', '/app/data/workflow.sqlite3', '--uploads', '/app/uploads',
                       '--destination', '/app/data/smoke-backup')
                docker('exec', api_name, 'python', 'scripts/product_backup.py', 'restore',
                       '--archive', '/app/data/smoke-backup', '--destination', '/app/data/smoke-restore')
                result['status'] = 'passed'
                result['checks'] = ['dashboard', 'authentication', 'manual_without_gpu_worker', 'durable_queue_after_api_restart',
                                    'idempotency', 'real_gpu_inference', 'original_hash', 'integrity',
                                    'worker_readiness', 'backup_restore']
                result['images'] = {kind: docker('image', 'inspect', '--format', '{{.Id}}',
                                                'cosmos-guardian-' + kind + ':dev') for kind in ('api', 'worker')}
                print(json.dumps({'status': result['status'], 'assessment_id': job['assessment_id'],
                                  'checks': result['checks'], 'output': str(args.output)}, ensure_ascii=False))
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        result['container_logs'] = {container: subprocess.run(
            ['docker', 'logs', '--tail=50', container], check=False, capture_output=True, text=True,
        ).stderr for container in containers}
        raise
    finally:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        for container in reversed(containers):
            subprocess.run(['docker', 'rm', '-f', container], check=False, capture_output=True)
        for volume in reversed(created_volumes):
            subprocess.run(['docker', 'volume', 'rm', volume], check=False, capture_output=True)


if __name__ == '__main__':
    main()
