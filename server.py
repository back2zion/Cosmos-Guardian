"""Analysis API and authenticated supervisor review workflow.

COSMOS_USERS maps private bearer tokens to {id, role}. No client-supplied identity.
COSMOS_INFERENCE_ENABLED=0 permits review without GPU dependencies or fake results.
"""
import hashlib
import json
import os
import secrets
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.audit import sha256_file
from core.jobs import JobStore, QueueFull
from core.reports import export_html, export_xlsx
from core.workflow import (
    CompletionRequest,
    Conflict,
    DecisionRequest,
    EditRequest,
    ReviewDecisionRequest,
    Revision,
    WorkflowError,
    WorkflowStore,
)


def create_app(*, agent=None, db_path=None, users=None, inference_enabled=None, upload_dir=None, lifespan=None):
    app = FastAPI(title="Cosmos Guardian API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("COSMOS_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
        allow_methods=["GET", "POST", "PUT"], allow_headers=["Authorization", "Content-Type", 'Idempotency-Key'],
    )
    configured_users = users if users is not None else json.loads(os.getenv("COSMOS_USERS", "{}"))
    identities = set()
    for token, user in configured_users.items():
        if (len(token) < 24 or token.startswith("REPLACE_") or not isinstance(user, dict)
                or not isinstance(user.get("id"), str) or not user["id"].strip()
                or user.get("role") not in ("operator", "supervisor")
                or user["id"] in identities):
            raise ValueError("COSMOS_USERS requires unique IDs, operator/supervisor roles and tokens of at least 24 characters")
        identities.add(user["id"])
    store = WorkflowStore(db_path or os.getenv("COSMOS_WORKFLOW_DB", "data/workflow.sqlite3"))
    app.state.store = store
    app.state.agent = agent
    enabled = inference_enabled if inference_enabled is not None else os.getenv("COSMOS_INFERENCE_ENABLED", "1") == "1"
    uploads = Path(upload_dir or os.getenv('COSMOS_UPLOAD_DIR', 'uploads'))
    jobs = JobStore(store, max_pending=int(os.getenv('COSMOS_MAX_PENDING_JOBS', '20')))
    app.state.jobs = jobs
    inference_lock = Lock()
    bearer = HTTPBearer(auto_error=False)

    def current_user(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]):
        if not configured_users:
            raise HTTPException(503, "서버에 사용자 계정이 설정되지 않았습니다. COSMOS_USERS를 설정하세요.")
        if credentials:
            for token, identity in configured_users.items():
                if secrets.compare_digest(token.encode(), credentials.credentials.encode()):
                    return {"id": identity["id"], "role": identity["role"]}
        raise HTTPException(401, "사용자 토큰을 확인하세요.", headers={"WWW-Authenticate": "Bearer"})

    User = Annotated[dict, Depends(current_user)]

    @app.exception_handler(WorkflowError)
    async def workflow_error(_, exc):
        code = 429 if isinstance(exc, QueueFull) else 409 if isinstance(exc, Conflict) else 422
        return JSONResponse(status_code=code, content={"detail": str(exc)})

    @app.exception_handler(PermissionError)
    async def permission_error(_, exc):
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    def get_document(assessment_id):
        try:
            return store.get(assessment_id)
        except KeyError:
            raise HTTPException(404, "평가서를 찾을 수 없습니다.") from None

    def validate_supervisor(supervisor_id):
        if not any(u["id"] == supervisor_id and u["role"] == "supervisor" for u in configured_users.values()):
            raise WorkflowError("설정된 관리감독자를 선택하세요.")

    @app.get("/health")
    def health():
        return {"status": "ok", "inference_enabled": enabled,
                "model_loaded": bool(app.state.agent and app.state.agent.model),
                "users_configured": bool(configured_users)}

    @app.get("/me")
    def me(actor: User):
        return actor

    @app.get('/ready')
    def ready():
        available = jobs.worker_available() if enabled else False
        is_ready = bool(configured_users) and (not enabled or available)
        return JSONResponse(status_code=200 if is_ready else 503, content={
            'ready': is_ready, 'review_available': bool(configured_users),
            'worker_available': available, 'inference_enabled': enabled,
            'product_stage': 'development-preview', 'site_validation': False,
        })

    def visible_job(job_id, actor):
        try:
            job = jobs.get(job_id)
        except KeyError:
            raise HTTPException(404, '작업을 찾을 수 없습니다.') from None
        if job['actor_id'] != actor['id'] and actor['role'] != 'supervisor':
            raise HTTPException(404, '작업을 찾을 수 없습니다.')
        return job

    @app.get('/jobs')
    def list_jobs(actor: User):
        return jobs.list(actor)

    @app.get('/jobs/{job_id}')
    def get_job(job_id: str, actor: User):
        return jobs.public(visible_job(job_id, actor))

    @app.post('/jobs/{job_id}/{command}')
    def control_job(job_id: str, command: str, actor: User):
        if command not in {'retry', 'cancel'}:
            raise HTTPException(404, '지원하지 않는 작업입니다.')
        if command == 'retry' and not enabled:
            raise HTTPException(503, '현재 추론이 비활성화되어 있습니다.')
        visible_job(job_id, actor)
        return jobs.control(job_id, actor, command)

    @app.post('/jobs', status_code=202)
    async def submit_job(actor: User, file: Annotated[UploadFile, File()],
                         idempotency_key: Annotated[str, Header(min_length=8, max_length=100)],
                         mode: Annotated[Literal['helmet', 'manual'], Form()] = 'helmet',
                         context: Annotated[str, Form(max_length=4000)] = ''):
        if not enabled and mode != 'manual':
            raise HTTPException(503, '현재 추론이 비활성화되어 있습니다.')
        filename = Path((file.filename or '').replace('\\', '/')).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise HTTPException(415, '현장 사진은 PNG·JPEG·WebP 이미지를 지원합니다.')
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f'{uuid4().hex}{suffix}'
        digest, size, keep = hashlib.sha256(), 0, False
        try:
            with path.open('wb') as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise HTTPException(413, '이미지는 최대 20 MiB까지 제출할 수 있습니다.')
                    destination.write(chunk)
                    digest.update(chunk)
            from PIL import Image, UnidentifiedImageError
            try:
                with Image.open(path) as picture:
                    if picture.format not in {'PNG', 'JPEG', 'WEBP'} or picture.width * picture.height > 20_000_000:
                        raise HTTPException(422, '지원 이미지 형식 또는 최대 2천만 화소 범위를 확인하세요.')
                    picture.verify()
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                raise HTTPException(422, '읽을 수 있는 이미지 파일이 아닙니다.') from None
            job, reused = jobs.submit({
                'input_name': filename, 'input_sha256': digest.hexdigest(), 'media_file': path.name,
                'media_type': 'image', 'bytes': size,
            }, context, actor, idempotency_key, profile='manual-v1' if mode == 'manual' else 'helmet-v1')
            keep = not reused
            return {**job, 'reused_request': reused}
        finally:
            await file.close()
            if not keep:
                path.unlink(missing_ok=True)

    @app.get("/supervisors")
    def supervisors(actor: User):
        return [{"id": u["id"]} for u in configured_users.values() if u["role"] == "supervisor"]

    @app.post("/analyze")
    async def analyze(actor: User, file: Annotated[UploadFile, File()], context: Annotated[str, Form()] = "General Safety"):
        if not enabled:
            raise HTTPException(503, "현재 서버는 검토 전용 모드입니다. 실제 모델 추론이 비활성화되어 있습니다.")
        if agent is None and os.getenv('COSMOS_LEGACY_STREAM_ENABLED', '0') != '1':
            raise HTTPException(410, '일반 안전 추론은 실험 기능입니다. 제품 분석은 /jobs를 사용하세요.')
        if len(context) > 4000:
            raise HTTPException(422, "분석 맥락은 4,000자 이하로 입력하세요.")
        filename = Path((file.filename or "").replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".webm", ".jpg", ".jpeg", ".png", ".webp"}:
            raise HTTPException(415, "지원하는 이미지 또는 영상 파일을 선택하세요.")
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"{uuid4().hex}{suffix}"
        digest, size = hashlib.sha256(), 0
        try:
            with path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 100 * 1024 * 1024:
                        raise HTTPException(413, "파일 크기는 100 MiB 이하여야 합니다.")
                    digest.update(chunk)
                    destination.write(chunk)
            if not size:
                raise HTTPException(422, "빈 파일은 분석할 수 없습니다.")
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

        def events():
            acquired = inference_lock.acquire(blocking=False)
            persisted = False
            try:
                if not acquired:
                    raise WorkflowError("다른 분석이 진행 중입니다. 완료 후 다시 요청하세요.")
                if app.state.agent is None:
                    from core.agent import CosmosGuardianAgent
                    app.state.agent = CosmosGuardianAgent(audit_log_path=os.getenv("COSMOS_AUDIT_LOG") or None)
                runtime = app.state.agent
                start = monotonic()
                for update in runtime.analyze_media(str(path.absolute()), safety_context=context):
                    if update["stage"] == "complete":
                        elapsed = round(monotonic() - start, 3)
                        document = store.create(update["result"], {
                            "input_name": filename, "input_sha256": digest.hexdigest(),
                            "media_file": path.name,
                            "media_type": "video" if suffix in {".mp4", ".avi", ".mov", ".webm"} else "image",
                            "model_id": runtime.model_id, "adapter_loaded": runtime.adapter_loaded,
                            "adapter": getattr(runtime, "_audit_adapter", None),
                            "label_grounding": runtime.use_label_grounding,
                            "context": context, "audit": update.get("audit"), "elapsed_seconds": elapsed,
                        }, actor)
                        persisted = True
                        update = {**update, "assessment_id": document["id"], "elapsed_seconds": elapsed}
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            except Exception as exc:  # noqa: BLE001 - convert inference failures into SSE error events
                yield f"data: {json.dumps({'stage': 'error', 'detail': str(exc)}, ensure_ascii=False)}\n\n"
            finally:
                if not persisted:
                    path.unlink(missing_ok=True)
                if acquired:
                    inference_lock.release()

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/assessments")
    def assessments(actor: User):
        return store.list()

    @app.get("/assessments/{assessment_id}")
    def assessment(assessment_id: str, actor: User):
        return get_document(assessment_id)

    @app.put("/assessments/{assessment_id}")
    def edit(assessment_id: str, body: EditRequest, actor: User):
        get_document(assessment_id)
        if body.form.supervisor_id:
            validate_supervisor(body.form.supervisor_id)
        return store.mutate(assessment_id, body.version, actor, "edit", form=body.form.model_dump(mode="json"))

    @app.post("/assessments/{assessment_id}/submit")
    def submit(assessment_id: str, body: Revision, actor: User):
        document = get_document(assessment_id)
        validate_supervisor(document["form"]["supervisor_id"])
        return store.mutate(assessment_id, body.version, actor, "submit")

    @app.post("/assessments/{assessment_id}/decisions/{decision}")
    def decide(assessment_id: str, decision: str, body: ReviewDecisionRequest, actor: User):
        get_document(assessment_id)
        if decision not in ("approve", "request_changes", "close"):
            raise HTTPException(404, "지원하지 않는 확인 작업입니다.")
        return store.mutate(assessment_id, body.version, actor, decision, note=body.note,
                            checklist=body.checks.model_dump() if body.checks else None)

    @app.post("/assessments/{assessment_id}/actions/{row_id}/complete")
    def complete_action(assessment_id: str, row_id: str, body: CompletionRequest, actor: User):
        get_document(assessment_id)
        return store.mutate(assessment_id, body.version, actor, "complete_action", row_id=row_id,
                            note=body.note, completion=body.model_dump(mode="json"))

    @app.post("/assessments/{assessment_id}/actions/{row_id}/{decision}")
    def verify_action(assessment_id: str, row_id: str, decision: str, body: DecisionRequest, actor: User):
        get_document(assessment_id)
        if decision not in ("verify", "reopen"):
            raise HTTPException(404, "지원하지 않는 조치 확인 작업입니다.")
        return store.mutate(assessment_id, body.version, actor, f"{decision}_action", row_id=row_id, note=body.note)

    @app.get("/assessments/{assessment_id}/integrity")
    def integrity(assessment_id: str, actor: User):
        get_document(assessment_id)
        return {"valid": store.verify(assessment_id)}

    @app.get("/assessments/{assessment_id}/media")
    def source_media(assessment_id: str, actor: User):
        document = get_document(assessment_id)
        if not store.verify(assessment_id):
            raise Conflict("변경 이력 무결성 확인에 실패했습니다.")
        filename = document["source"].get("media_file", "")
        if not filename or Path(filename).name != filename:
            raise HTTPException(404, "원본 미디어가 보관되지 않은 기록입니다.")
        path = uploads / filename
        if not path.is_file():
            raise HTTPException(404, "원본 미디어를 찾을 수 없습니다.")
        if sha256_file(path) != document["source"]["input_sha256"]:
            raise Conflict("원본 미디어의 해시가 분석 당시와 다릅니다.")
        return FileResponse(path, filename=document["source"]["input_name"],
                            headers={"Cache-Control": "no-store"})

    @app.get("/assessments/{assessment_id}/export/{format}")
    def export(assessment_id: str, format: str, actor: User):
        document = get_document(assessment_id)
        if not store.verify(assessment_id):
            raise Conflict("변경 이력 무결성 확인에 실패했습니다.")
        if format == "xlsx":
            data, media_type = export_xlsx(document), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif format == "html":
            data, media_type = export_html(document), "text/html"
        else:
            raise HTTPException(404, "지원하지 않는 출력 형식입니다.")
        return Response(data, media_type=media_type, headers={
            "Content-Disposition": f'attachment; filename="risk-assessment-{assessment_id}-v{document["version"]}.{format}"',
            "Cache-Control": "no-store",
        })

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8888)
