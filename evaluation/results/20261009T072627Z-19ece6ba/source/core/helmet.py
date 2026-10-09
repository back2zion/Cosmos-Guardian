"""Strict, deliberately narrow helmet observation contract. No site safety score."""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HelmetObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    decision: Literal['violation', 'clear', 'unknown']
    evidence: str = Field(min_length=1, max_length=600)


def parse_observation(text):
    # No bracket repair, truncated-object acceptance, or fabricated safe fallback.
    observation = HelmetObservation.model_validate(json.loads(text.strip()))
    return {
        'profile': 'helmet-v1', 'assessment_status': observation.decision,
        'scope': '안전모 착용 의무 구역의 이미지 판별만 포함 · 전체 현장 안전 판단 아님',
        'evidence': observation.evidence, 'requires_human_review': True,
        'hazards_detected': [{
            'type': '안전모 미착용 의심', 'event_type': 'ppe_violation',
            'severity': 'NeedsReview', 'description': observation.evidence,
        }] if observation.decision == 'violation' else [],
        'recommendation': '원본에서 착용 여부와 현장 착용 기준을 확인하세요. 가능성·중대성은 담당자가 평가합니다.',
    }


def parse_classification(text):
    answer = text.strip().upper().rstrip('.')
    decisions = {'YES': 'violation', 'NO': 'clear', 'UNKNOWN': 'unknown'}
    if answer not in decisions:
        raise ValueError('Expected exactly YES, NO or UNKNOWN')
    result = parse_observation(json.dumps({
        'decision': decisions[answer],
        'evidence': '이미지 단위 판별 결과입니다. 사람별 위치·검출 박스는 제공하지 않으므로 원본을 직접 확인하세요.',
    }))
    result['model_answer'] = answer
    result['evidence_kind'] = 'image_classification_without_localization'
    return result
