import json

import pytest

from core.evaluation import load_manifest
from scripts.benchmark import evaluate, preflight
from scripts.prepare_public_ppe_eval import classify, prepare


def sample(name, labels):
    return {'filepath': f'data/{name}.png', 'ground_truth': {'detections': [{'label': label} for label in labels]}}


def test_ppe_targets_come_from_original_annotations():
    assert classify(sample('a', ['head', 'helmet'])) is True
    assert classify(sample('b', ['helmet', 'helmet'])) is False
    assert classify(sample('c', ['person', 'helmet'])) is None
    assert classify(sample('d', [])) is None
    difficult = sample('e', ['head'])
    difficult['ground_truth']['detections'][0]['difficult'] = 1
    assert classify(difficult) is None


def test_download_selection_hashes_and_scope_are_reproducible(tmp_path, monkeypatch):
    upstream = {'samples': [sample('a', ['head']), sample('b', ['helmet']), sample('c', ['head']), sample('d', ['helmet'])]}
    def fetch(filename, revision):
        assert revision == 'test-revision'
        return json.dumps(upstream).encode() if filename == 'samples.json' else filename.encode()
    monkeypatch.setattr('scripts.prepare_public_ppe_eval.fetch', fetch)
    destination = tmp_path / 'manifest.json'
    first = prepare(destination, tmp_path / 'media', per_class=1, revision='test-revision')
    second = prepare(destination, tmp_path / 'media', per_class=1, revision='test-revision')
    assert first == second
    assert {s['is_hazardous'] for s in first['samples']} == {True, False}
    assert len({s['group_id'] for s in first['samples']}) == 1  # No invented source-session independence.
    records = load_manifest(destination)
    (tmp_path / records[0]['media_path']).write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Media hash mismatch'):
        load_manifest(destination)


def test_benchmark_uses_one_shared_scope_without_ground_truth(tmp_path, monkeypatch):
    upstream = {'samples': [sample('a', ['head']), sample('b', ['helmet'])]}
    monkeypatch.setattr('scripts.prepare_public_ppe_eval.fetch', lambda filename, revision:
                        json.dumps(upstream).encode() if filename == 'samples.json' else filename.encode())
    manifest = tmp_path / 'manifest.json'
    data = prepare(manifest, tmp_path / 'media', per_class=1)
    class Args:
        base_only = True
        training_manifest = None
        adapter = tmp_path
    args = Args()
    args.manifest = manifest
    check, records = preflight(args)
    assert check['safety_context'] == data['safety_context']
    assert check['scope'] == data['scope']
    contexts = []
    class Agent:
        def analyze_media(self, media_path, safety_context):
            contexts.append(safety_context)
            yield {'stage': 'complete', 'result': {'hazards_detected': []}}
    evaluate(Agent(), records, check['safety_context'])
    assert contexts == [data['safety_context']] * 2


def test_new_splits_exclude_both_prior_ids_and_duplicate_media(tmp_path, monkeypatch):
    upstream = {'samples': [sample(name, [label]) for name, label in [
        ('prior', 'head'), ('same-bytes-new-id', 'head'), ('new-positive', 'head'),
        ('old-negative', 'helmet'), ('new-negative', 'helmet')]]}
    import hashlib
    prior = tmp_path / 'prior.json'
    prior.write_text(json.dumps({'samples': [
        {'id': 'public-ppe:prior', 'media_sha256': hashlib.sha256(b'duplicate').hexdigest()},
        {'id': 'public-ppe:old-negative', 'media_sha256': hashlib.sha256(b'old-negative').hexdigest()},
    ]}))
    def fetch(filename, revision):
        if filename == 'samples.json':
            return json.dumps(upstream).encode()
        return b'duplicate' if filename == 'data/same-bytes-new-id.png' else filename.encode()
    monkeypatch.setattr('scripts.prepare_public_ppe_eval.fetch', fetch)
    result = prepare(tmp_path / 'test.json', tmp_path / 'media', per_class=1, exclude_manifests=[prior])
    assert {item['id'] for item in result['samples']} == {'public-ppe:new-positive', 'public-ppe:new-negative'}
    assert result['excluded_manifests'][0]['sha256'] == hashlib.sha256(prior.read_bytes()).hexdigest()
