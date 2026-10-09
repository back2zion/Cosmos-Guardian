"""Durable single-site job queue sharing the assessment transaction database.

Claim tokens fence expired workers. Completing a job and creating its assessment
are one transaction. Idempotency is actor-scoped, not a cross-user result cache.
"""
import hashlib
import json
import time
from uuid import uuid4

from .workflow import Conflict, WorkflowError, WorkflowStore, canonical, now


class QueueFull(WorkflowError):
    pass


class JobStore:
    def __init__(self, workflow: WorkflowStore, max_pending=20, lease_seconds=30):
        self.workflow = workflow
        self.max_pending = max_pending
        self.lease_seconds = lease_seconds
        with workflow.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, request_key TEXT NOT NULL,
                    state TEXT NOT NULL, created REAL NOT NULL, body TEXT NOT NULL,
                    UNIQUE(actor_id, request_key)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(state, created);
                CREATE TABLE IF NOT EXISTS workers (id TEXT PRIMARY KEY, heartbeat REAL NOT NULL);
            ''')

    def _load(self, db, job_id):
        row = db.execute('SELECT body FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row[0])

    def _save(self, db, job):
        job['updated_at'] = now()
        db.execute('UPDATE jobs SET state=?,body=? WHERE id=?', (job['state'], canonical(job), job['id']))

    @staticmethod
    def public(job):
        return {k: v for k, v in job.items() if k not in {'claim', 'lease_until', 'fingerprint', 'actor', 'request_key'}}

    def get(self, job_id):
        with self.workflow.connect() as db:
            return self._load(db, job_id)

    def list(self, actor):
        with self.workflow.connect() as db:
            query = 'SELECT body FROM jobs' + ('' if actor['role'] == 'supervisor' else ' WHERE actor_id=?')
            rows = db.execute(query + ' ORDER BY created DESC LIMIT 100',
                              () if actor['role'] == 'supervisor' else (actor['id'],))
            return [self.public(json.loads(r[0])) for r in rows]

    def submit(self, source, context, actor, request_key, profile='helmet-v1'):
        if profile not in {'helmet-v1', 'manual-v1'}:
            raise WorkflowError('지원하지 않는 작성 방식입니다.')
        manual = profile == 'manual-v1'
        fingerprint = hashlib.sha256(canonical({
            'sha256': source['input_sha256'], 'context': context, 'profile': profile,
        }).encode()).hexdigest()
        with self.workflow.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT body FROM jobs WHERE actor_id=? AND request_key=?',
                               (actor['id'], request_key)).fetchone()
            if prior:
                job = json.loads(prior[0])
                if fingerprint != job['fingerprint']:
                    raise Conflict('같은 요청 키로 다른 자료를 제출할 수 없습니다.')
                job['duplicate_submissions'] += 1
                self._save(db, job)
                return self.public(job), True
            if not manual and db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= self.max_pending:
                raise QueueFull('분석 대기열이 가득 찼습니다. 잠시 후 다시 제출하세요.')
            job = {'id': uuid4().hex, 'actor': actor, 'actor_id': actor['id'], 'state': 'queued',
                   'created_at': now(), 'updated_at': now(), 'source': source, 'context': context,
                   'profile': profile, 'request_key': request_key, 'fingerprint': fingerprint,
                   'attempt': 0, 'attempts': [], 'duplicate_submissions': 0,
                   'assessment_id': None, 'error': None, 'stage': 'queued'}
            if manual:
                self.workflow.create({
                    'profile': 'manual-v1', 'analysis_performed': False, 'assessment_status': 'unknown',
                    'requires_human_review': True, 'hazards_detected': [],
                    'scope': 'AI 분석 없이 직접 작성하는 평가서입니다. 위험요인이 자동 검출되지 않았습니다.',
                    'evidence': '원본 사진과 실제 현장을 확인하고 위험요인을 직접 등록하세요.',
                    'recommendation': '현장 기준에 따라 위험요인·가능성·중대성·조치를 직접 평가하세요.',
                }, {**source, 'context': context, 'job_id': job['id'], 'profile': 'manual-v1',
                    'model_id': None, 'analysis_performed': False}, actor,
                    assessment_id=job['id'], transaction=db)
                job.update(state='completed', stage='manual_draft', assessment_id=job['id'])
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)',
                       (job['id'], actor['id'], request_key, job['state'], time.time(), canonical(job)))
            return self.public(job), False

    def control(self, job_id, actor, command):
        with self.workflow.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self._load(db, job_id)
            if job['actor_id'] != actor['id']:
                raise PermissionError('본인이 제출한 작업만 취소하거나 재시도할 수 있습니다.')
            if command == 'cancel':
                if job['state'] not in {'queued', 'running'}:
                    raise Conflict('대기 또는 실행 중인 작업만 취소할 수 있습니다.')
                job.update(state='cancelled', stage='cancelled', error=None)
            elif command == 'retry':
                if job['state'] not in {'failed', 'cancelled'}:
                    raise Conflict('실패하거나 취소된 작업만 재시도할 수 있습니다.')
                if job['attempt'] >= 3:
                    raise Conflict('최대 3회 시도했습니다. 자료를 확인한 뒤 새 작업으로 제출하세요.')
                if db.execute("SELECT count(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= self.max_pending:
                    raise QueueFull('분석 대기열이 가득 찼습니다.')
                job.update(state='queued', stage='queued', error=None)
            else:
                raise WorkflowError('지원하지 않는 작업입니다.')
            self._save(db, job)
            return self.public(job)

    def pulse(self, worker_id):
        with self.workflow.connect() as db:
            db.execute('INSERT OR REPLACE INTO workers VALUES (?,?)', (worker_id, time.time()))

    def worker_available(self):
        with self.workflow.connect() as db:
            row = db.execute('SELECT max(heartbeat) FROM workers').fetchone()
            return bool(row[0] and row[0] > time.time() - 20)

    def claim(self):
        with self.workflow.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for row in db.execute("SELECT body FROM jobs WHERE state='running'").fetchall():
                stale = json.loads(row[0])
                if stale['lease_until'] < time.time():
                    stale.update(state='failed', stage='failed', error='worker_lost')
                    stale['attempts'].append({'attempt': stale['attempt'], 'outcome': 'worker_lost', 'at': now()})
                    self._save(db, stale)
            row = db.execute("SELECT body FROM jobs WHERE state='queued' ORDER BY created,id LIMIT 1").fetchone()
            if row is None:
                return None
            job = json.loads(row[0])
            job.update(state='running', stage='starting', claim=uuid4().hex,
                       lease_until=time.time() + self.lease_seconds, started_at=now(), attempt=job['attempt'] + 1)
            self._save(db, job)
            return job

    def heartbeat(self, job_id, claim, stage=None):
        with self.workflow.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self._load(db, job_id)
            if job['state'] != 'running' or job.get('claim') != claim:
                return False
            job['lease_until'] = time.time() + self.lease_seconds
            if stage:
                job['stage'] = stage
            self._save(db, job)
            return True

    def finish(self, job_id, claim, result, metadata, elapsed):
        with self.workflow.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self._load(db, job_id)
            if job['state'] != 'running' or job.get('claim') != claim or job['lease_until'] < time.time():
                raise Conflict('취소되거나 소유권을 잃은 작업의 결과는 저장하지 않습니다.')
            document = self.workflow.create(result, {**job['source'], **metadata, 'job_id': job_id,
                                                      'context': job['context'], 'elapsed_seconds': round(elapsed, 3)},
                                            job['actor'], assessment_id=job_id, transaction=db)
            job.update(state='completed', stage='completed', assessment_id=document['id'], error=None)
            job['attempts'].append({'attempt': job['attempt'], 'outcome': 'completed',
                                    'elapsed_seconds': round(elapsed, 3), 'at': now()})
            self._save(db, job)
            return self.public(job)

    def fail(self, job_id, claim, code, elapsed):
        with self.workflow.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            job = self._load(db, job_id)
            if job.get('claim') != claim or job['state'] not in {'running', 'cancelled'}:
                return
            outcome = 'cancelled' if job['state'] == 'cancelled' else code
            job.update(state='cancelled' if outcome == 'cancelled' else 'failed', stage=outcome, error=outcome)
            job['attempts'].append({'attempt': job['attempt'], 'outcome': outcome,
                                    'elapsed_seconds': round(elapsed, 3), 'at': now()})
            self._save(db, job)
