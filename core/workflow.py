"""Persistent, versioned supervisor review and corrective-action workflow.

Risk scores use an explicitly selected project 5x5 method. They are not the
model's safety score, a prescribed statutory form, or a compliance decision.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class WorkflowError(ValueError):
    pass


class Conflict(WorkflowError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RiskRow(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    hazard: str = Field(min_length=1, max_length=4000)
    consequence: str = Field(default="", max_length=4000)
    existing_controls: str = Field(default="", max_length=4000)
    likelihood: int | None = Field(default=None, ge=1, le=5, strict=True)
    severity: int | None = Field(default=None, ge=1, le=5, strict=True)
    additional_controls: str = Field(default="", max_length=4000)
    owner: str = Field(default="", max_length=200)
    due_date: date | None = None


class AssessmentForm(StrictModel):
    site: str = Field(default="", max_length=300)
    process: str = Field(default="", max_length=300)
    task: str = Field(default="", max_length=1000)
    assessment_date: date | None = None
    participants: str = Field(default="", max_length=2000)
    supervisor_id: str = Field(default="", max_length=200)
    acceptance_criteria: str = Field(default="", max_length=2000)
    review_note: str = Field(default="", max_length=4000)
    rows: list[RiskRow] = Field(default_factory=list, max_length=100)


class Revision(StrictModel):
    version: int = Field(ge=1, strict=True)


class EditRequest(Revision):
    form: AssessmentForm


class DecisionRequest(Revision):
    note: str = Field(min_length=1, max_length=4000)


class ReviewChecklist(StrictModel):
    source_checked: bool = Field(strict=True)
    hazards_checked: bool = Field(strict=True)
    controls_checked: bool = Field(strict=True)
    criteria_checked: bool = Field(strict=True)


class ReviewDecisionRequest(DecisionRequest):
    checks: ReviewChecklist | None = None


class CompletionRequest(DecisionRequest):
    evidence: str = Field(min_length=1, max_length=4000)
    residual_likelihood: int = Field(ge=1, le=5, strict=True)
    residual_severity: int = Field(ge=1, le=5, strict=True)


def validate_analysis(result):
    if not isinstance(result, dict) or result.get("error"):
        raise WorkflowError("분석 실패 결과는 평가서로 등록할 수 없습니다.")
    hazards = result.get("hazards_detected")
    if not isinstance(hazards, list) or len(hazards) > 100:
        raise WorkflowError("분석 결과에 유효한 hazards_detected 목록이 필요합니다.")
    for hazard in hazards:
        if not isinstance(hazard, dict) or not isinstance(hazard.get("type"), str) or not hazard["type"].strip():
            raise WorkflowError("분석 결과의 위험 유형이 누락되었습니다.")
        for field in ("description", "dynamic_prediction"):
            if not isinstance(hazard.get(field, ""), str):
                raise WorkflowError("분석 결과의 설명 형식이 올바르지 않습니다.")
    if not isinstance(result.get("recommendation", ""), str):
        raise WorkflowError("분석 결과의 권고사항 형식이 올바르지 않습니다.")


def validate_submission(form):
    for key in ("site", "process", "task", "assessment_date", "participants", "supervisor_id", "acceptance_criteria", "review_note"):
        if not form[key]:
            raise WorkflowError(f"검토 요청 전에 {key} 항목을 입력해야 합니다.")
    for row in form["rows"]:
        for key in ("consequence", "existing_controls", "likelihood", "severity", "additional_controls", "owner", "due_date"):
            if not row[key]:
                raise WorkflowError(f"위험요인 '{row['hazard']}': {key} 항목을 입력해야 합니다.")


class WorkflowStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS assessments (id TEXT PRIMARY KEY, document TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    assessment_id TEXT NOT NULL, seq INTEGER NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY (assessment_id, seq)
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _load(self, db, assessment_id):
        row = db.execute("SELECT document FROM assessments WHERE id=?", (assessment_id,)).fetchone()
        if row is None:
            raise KeyError(assessment_id)
        return json.loads(row[0])

    def get(self, assessment_id):
        with self.connect() as db:
            document = self._load(db, assessment_id)
            document["history"] = [json.loads(r[0]) for r in db.execute(
                "SELECT body FROM events WHERE assessment_id=? ORDER BY seq", (assessment_id,)
            )]
            return document

    def list(self):
        with self.connect() as db:
            documents = [json.loads(row[0]) for row in db.execute("SELECT document FROM assessments")]
        return [{key: d[key] for key in ("id", "version", "status", "created_at", "updated_at", "source", "form")}
                for d in sorted(documents, key=lambda d: d["updated_at"], reverse=True)]

    def _write(self, db, document, actor, event_type, note=""):
        last = db.execute("SELECT seq, body FROM events WHERE assessment_id=? ORDER BY seq DESC LIMIT 1", (document["id"],)).fetchone()
        document["updated_at"] = now()
        event = {
            "seq": last[0] + 1 if last else 0, "assessment_id": document["id"],
            "timestamp": document["updated_at"], "actor": actor, "type": event_type,
            "note": note, "snapshot": document,
            "prev_hash": json.loads(last[1])["hash"] if last else "0" * 64,
        }
        event["hash"] = hashlib.sha256(canonical(event).encode()).hexdigest()
        db.execute("INSERT INTO events VALUES (?, ?, ?)", (document["id"], event["seq"], canonical(event)))
        db.execute("INSERT OR REPLACE INTO assessments VALUES (?, ?)", (document["id"], canonical(document)))

    def create(self, result, source, actor, *, assessment_id=None, transaction=None):
        validate_analysis(result)
        rows = [RiskRow(
            id=uuid4().hex, hazard=h["type"],
            consequence=h.get("dynamic_prediction", "") or h.get("description", ""),
            additional_controls=result.get("recommendation", ""),
        ) for h in result["hazards_detected"]]
        document = {
            "id": assessment_id or uuid4().hex, "version": 1, "status": "draft", "created_at": now(),
            "created_by": actor["id"], "source": source, "original_result": result,
            "form": AssessmentForm(rows=rows).model_dump(mode="json"),
            "approval": None, "actions": {},
            "method": "project-5x5-v1: likelihood (1–5) × severity (1–5); human-entered",
            "template_version": "cosmos-risk-assessment-v1",
        }
        if transaction is not None:
            self._write(transaction, document, actor, 'created')
            return document
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._write(db, document, actor, "created")
        return self.get(document["id"])

    @staticmethod
    def _require_supervisor(document, actor):
        if actor["role"] != "supervisor" or actor["id"] != document["form"]["supervisor_id"]:
            raise PermissionError("지정된 관리감독자만 확인할 수 있습니다.")

    def mutate(self, assessment_id, version, actor, command, *, form=None, note="", row_id=None, completion=None, checklist=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            document = self._load(db, assessment_id)
            if document["version"] != version:
                raise Conflict("다른 사용자가 변경했습니다. 최신 평가서를 다시 불러오세요.")
            if not self._verify(db, assessment_id):
                raise Conflict("변경 이력 무결성 확인에 실패했습니다.")
            status = document["status"]
            if command == "edit":
                if status not in ("draft", "changes_requested"):
                    raise Conflict("초안 또는 보완 요청 상태에서만 수정할 수 있습니다.")
                form = AssessmentForm.model_validate(form).model_dump(mode="json")
                ids = [r["id"] for r in form["rows"]]
                if len(ids) != len(set(ids)):
                    raise WorkflowError("위험요인 ID가 중복되었습니다.")
                document["form"] = form
            elif command == "submit":
                if status not in ("draft", "changes_requested"):
                    raise Conflict("검토 요청 가능한 상태가 아닙니다.")
                validate_submission(document["form"])
                document["status"] = "pending_review"
            elif command in ("approve", "request_changes"):
                self._require_supervisor(document, actor)
                if status != "pending_review":
                    raise Conflict("검토 대기 상태에서만 승인 또는 보완 요청할 수 있습니다.")
                if not note.strip():
                    raise WorkflowError("검토 의견이 필요합니다.")
                if command == "approve":
                    validate_submission(document["form"])
                    if not checklist or set(checklist) != set(ReviewChecklist.model_fields) or not all(value is True for value in checklist.values()):
                        raise WorkflowError("원본, 위험요인, 개선조치, 평가 기준을 모두 확인해야 승인할 수 있습니다.")
                    document["status"] = "approved"
                    document["approval"] = {"actor": actor["id"], "timestamp": now(), "note": note,
                                            "reviewed_version": version, "checks": checklist}
                    document["actions"] = {r["id"]: {"status": "open"} for r in document["form"]["rows"]}
                else:
                    document["status"] = "changes_requested"
            elif command in ("complete_action", "verify_action", "reopen_action"):
                if status != "approved" or row_id not in document["actions"]:
                    raise Conflict("승인된 평가서의 조치 항목이 아닙니다.")
                action = document["actions"][row_id]
                if command == "complete_action":
                    if action["status"] != "open":
                        raise Conflict("미완료 조치만 완료 등록할 수 있습니다.")
                    details = CompletionRequest.model_validate(completion).model_dump(mode="json")
                    action.update({k: v for k, v in details.items() if k != "version"})
                    action.update(status="completed", completed_by=actor["id"], completed_at=now())
                else:
                    self._require_supervisor(document, actor)
                    if action["status"] != "completed" or not note.strip():
                        raise Conflict("완료 등록된 조치와 확인 의견이 필요합니다.")
                    if command == "verify_action":
                        action.update(status="verified", verified_by=actor["id"], verified_at=now(), verification_note=note)
                    else:
                        document["actions"][row_id] = {"status": "open", "reopen_note": note}
            elif command == "close":
                self._require_supervisor(document, actor)
                if status != "approved" or any(a["status"] != "verified" for a in document["actions"].values()):
                    raise Conflict("모든 개선조치를 재확인한 후 종결할 수 있습니다.")
                if not note.strip():
                    raise WorkflowError("종결 의견이 필요합니다.")
                document["status"] = "closed"
                document["closure"] = {"actor": actor["id"], "timestamp": now(), "note": note}
            else:
                raise WorkflowError("지원하지 않는 작업입니다.")
            document["version"] += 1
            self._write(db, document, actor, command, note)
        return self.get(assessment_id)

    def _verify(self, db, assessment_id):
        try:
            return self._verify_chain(db, assessment_id)
        except (KeyError, TypeError, ValueError):
            return False

    def _verify_chain(self, db, assessment_id):
        previous = "0" * 64
        snapshot = None
        for seq, row in enumerate(db.execute("SELECT body FROM events WHERE assessment_id=? ORDER BY seq", (assessment_id,))):
            event = json.loads(row[0])
            digest = event.pop("hash")
            if (event["seq"] != seq or event["assessment_id"] != assessment_id
                    or event["snapshot"]["version"] != seq + 1
                    or event["prev_hash"] != previous
                    or hashlib.sha256(canonical(event).encode()).hexdigest() != digest):
                return False
            previous, snapshot = digest, event["snapshot"]
        return snapshot is not None and snapshot == self._load(db, assessment_id)

    def verify(self, assessment_id):
        with self.connect() as db:
            self._load(db, assessment_id)
            try:
                return self._verify(db, assessment_id)
            except (KeyError, TypeError, ValueError):
                return False
