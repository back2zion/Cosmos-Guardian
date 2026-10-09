import argparse
import json
from pathlib import Path

import pytest

from core.evaluation import load_manifest, prediction, summarize, validate_holdout
from scripts.benchmark import evaluate, preflight


@pytest.fixture
def manifest(tmp_path):
    (tmp_path / 'hazard.png').write_bytes(b'hazard')
    (tmp_path / 'safe.png').write_bytes(b'safe')
    path = tmp_path / 'test.json'
    items = [
        {'id': 'test-hazard', 'group_id': 'site-a-shift-1', 'media_path': 'hazard.png', 'is_hazardous': True,
         'events': ['entrapment'], 'annotation_source': 'test-only annotation'},
        {'id': 'test-safe', 'group_id': 'site-b-shift-2', 'media_path': 'safe.png', 'is_hazardous': False,
         'events': [], 'annotation_source': 'test-only annotation'},
    ]
    path.write_text(json.dumps({'samples': items}))
    return path


def test_holdout_requires_safe_samples_and_excludes_training(manifest):
    samples = load_manifest(manifest)
    validate_holdout(samples, set())
    with pytest.raises(ValueError, match='overlap'):
        validate_holdout(samples, {'test-hazard'})
    for training in [dict(samples[0], id='different'), dict(samples[0], id='different', group_id='different')]:
        with pytest.raises(ValueError, match='overlap'):
            validate_holdout(samples, set(), [training])
    with pytest.raises(ValueError, match='both hazardous and safe'):
        validate_holdout(samples[:1], set())


@pytest.mark.parametrize('change', [
    {'is_hazardous': 'false'}, {'events': []}, {'events': ['unknown']},
    {'annotation_source': ''}, {'media_path': 'missing.png'},
])
def test_manifest_rejects_invalid_labels_or_missing_media(manifest, change):
    data = json.loads(manifest.read_text())
    data['samples'][0].update(change)
    manifest.write_text(json.dumps(data))
    with pytest.raises((ValueError, TypeError)):
        load_manifest(manifest)


def test_manifest_rejects_duplicate_content(manifest):
    data = json.loads(manifest.read_text())
    data['samples'][1]['media_path'] = 'hazard.png'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate media'):
        load_manifest(manifest)


def test_metrics_distinguish_false_alarms_misses_and_failures():
    records = [
        {'is_hazardous': True, 'hazard_detected': True, 'events': ['fall'], 'predicted_events': ['fall'], 'latency_seconds': 3},
        {'is_hazardous': True, 'hazard_detected': False, 'events': ['fall'], 'predicted_events': [], 'latency_seconds': 5},
        {'is_hazardous': False, 'hazard_detected': True, 'events': [], 'predicted_events': ['fall'], 'latency_seconds': 7},
        {'is_hazardous': False, 'hazard_detected': False, 'events': [], 'predicted_events': [], 'latency_seconds': 9},
        {'is_hazardous': True, 'error': 'inference failure'},
        {'is_hazardous': False, 'error': 'parse failure'},
    ]
    summary = summarize(records)
    assert summary['confusion'] == {'tp': 1, 'fp': 1, 'tn': 1, 'fn': 1, 'invalid_hazardous': 1, 'invalid_safe': 1}
    assert summary['precision'] == 0.5
    assert summary['recall_valid_outputs'] == 0.5
    assert summary['effective_recall_including_failures'] == 0.333333
    assert summary['false_positive_rate_valid_outputs'] == 0.5
    assert summary['mean_latency_seconds'] == 6
    assert summary['event_exact_match_rate'] == 0.333333


def test_errors_do_not_produce_zero_latency_or_safe_predictions():
    summary = summarize([{'is_hazardous': False, 'error': 'OOM'}])
    assert summary['mean_latency_seconds'] is None
    assert summary['precision'] is None
    assert summary['false_positive_rate_valid_outputs'] is None
    assert summary['confusion']['tn'] == 0


def test_prediction_uses_structured_hazards_not_keywords():
    assert prediction({'hazards_detected': [], 'recommendation': 'No risk; avoid collision'}) == (False, [])
    assert prediction({'hazards_detected': [{'type': 'fall'}]}) == (True, ['unclassified'])
    with pytest.raises((ValueError, TypeError)):
        prediction({'recommendation': 'safe'})


def test_missing_adapter_fails_preflight_without_loading_models(manifest, tmp_path):
    args = argparse.Namespace(manifest=manifest, training_manifest=None, base_only=False, adapter=tmp_path / 'absent')
    check, _ = preflight(args)
    assert not check['ready']
    assert any('Missing LoRA artifact' in error for error in check['errors'])
    assert any('training manifest' in error for error in check['errors'])


def test_evaluation_continues_after_sample_error(manifest):
    class Agent:
        def analyze_media(self, media_path, safety_context):
            if Path(media_path).stem == 'hazard':
                raise RuntimeError('inference failed')
            yield {'stage': 'complete', 'result': {'hazards_detected': []}}
    records = evaluate(Agent(), load_manifest(manifest))
    assert len(records) == 2
    assert 'error' in records[0] and 'hazard_detected' not in records[0]
    assert records[1]['hazard_detected'] is False
