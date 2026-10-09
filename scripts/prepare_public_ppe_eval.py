"""Prepare a deterministic public helmet-compliance evaluation, never model labels.

The upstream bounding boxes supply the target: any `head` means missing helmet;
helmet-only images are the negative class. This does not label overall site safety.
"""
import argparse
import hashlib
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

REPO = 'Voxel51/hard-hat-detection'
REVISION = 'd11e7d6b69ba1a22422db3230c0415d39a699d7c'
CONTEXT = (
    'Evaluate ONLY safety helmet compliance for visible people. '
    'Report a ppe_violation hazard when at least one visible head lacks a safety helmet. '
    'If all visible heads wear safety helmets, return hazards_detected as an empty list. '
    'Exclude other equipment, machinery, fall, electrical or site hazards from this assessment. '
    'This assessment does not determine overall site safety.'
)
ROOT = Path(__file__).resolve().parents[1]


def fetch(filename, revision):
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{filename}'
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def classify(sample):
    detections = sample.get('ground_truth', {}).get('detections', [])
    counts = Counter(d.get('label') for d in detections)
    if not counts or set(counts) - {'head', 'helmet'}:
        return None
    if any(d.get('difficult', 0) for d in detections):
        return None
    return bool(counts['head'])


def prepare(output, media_dir, per_class=50, seed=42, revision=REVISION, exclude_manifests=()):
    if per_class < 1:
        raise ValueError('per_class must be positive')
    excluded_ids, excluded_hashes, exclusions = set(), set(), []
    for path in exclude_manifests:
        content = Path(path).read_bytes()
        prior = json.loads(content)
        excluded_ids.update(s['id'] for s in prior['samples'])
        excluded_hashes.update(s['media_sha256'] for s in prior['samples'])
        exclusions.append({'manifest': Path(path).name, 'sha256': hashlib.sha256(content).hexdigest()})
    metadata = fetch('samples.json', revision)
    upstream = json.loads(metadata)['samples']
    candidates = {True: [], False: []}
    for sample in upstream:
        label = classify(sample)
        if label is not None:
            candidates[label].append(sample)
    rng = random.Random(seed)
    for pool in candidates.values():
        pool.sort(key=lambda sample: sample['filepath'])
        rng.shuffle(pool)
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir.parent / 'upstream_samples.json').write_bytes(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes, records = set(), []
    for hazardous in (True, False):
        selected = 0
        for sample in candidates[hazardous]:
            source_path = PurePosixPath(sample['filepath'])
            if f'public-ppe:{source_path.stem}' in excluded_ids:
                continue
            if source_path.parts[0] != 'data' or len(source_path.parts) != 2 or source_path.suffix != '.png':
                raise ValueError(f'Unexpected upstream path: {source_path}')
            content = fetch(str(source_path), revision)
            digest = hashlib.sha256(content).hexdigest()
            if digest in seen_hashes or digest in excluded_hashes:
                continue
            seen_hashes.add(digest)
            destination = media_dir / source_path.name
            temporary = destination.with_suffix('.part')
            temporary.write_bytes(content)
            temporary.replace(destination)
            # os.path.relpath also handles an output manifest outside the media folder.
            import os
            counts = Counter(d['label'] for d in sample['ground_truth']['detections'])
            records.append({
                'id': f'public-ppe:{source_path.stem}',
                'group_id': f'{REPO}:source-session-unknown',
                'media_path': os.path.relpath(destination.resolve(), output.parent.resolve()),
                'media_sha256': digest,
                'is_hazardous': hazardous, 'events': ['ppe_violation'] if hazardous else [],
                'annotation_source': f'{REPO}@{revision}:samples.json#ground_truth; image-level aggregation of upstream labels',
                'upstream_path': str(source_path), 'upstream_object_counts': dict(counts),
            })
            selected += 1
            print(f"{'missing-helmet' if hazardous else 'helmet-only'} {selected}/{per_class}: {source_path.name}", flush=True)
            if selected == per_class:
                break
        if selected != per_class:
            raise ValueError('Insufficient distinct images for requested balanced sample')
    rng.shuffle(records)
    manifest = {
        'name': 'public-ppe-helmet-compliance', 'source': f'https://huggingface.co/datasets/{REPO}',
        'source_revision': revision, 'source_metadata_sha256': hashlib.sha256(metadata).hexdigest(),
        'license_as_declared_by_source': 'CC0-1.0', 'seed': seed, 'samples_per_class': per_class,
        'safety_context': CONTEXT,
        'scope': 'Helmet presence only; negative means no unprotected head annotated, not a safe worksite',
        'limitations': [
            'Targets are upstream annotations, not independent review by an industrial safety expert.',
            'Original session groups are unavailable; all samples share one conservative unknown-session group.',
            'Exact duplicate files are removed; near-duplicates and pretraining overlap are not verified.',
            'Only base-model evaluation is immediately available; no project LoRA weights are supplied.',
            'The upstream dataset calls its only split train; this project does not train on this selected evaluation subset.',
        ],
        'samples': records,
    }
    if exclusions:
        manifest['excluded_manifests'] = exclusions
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Prepared {len(records)} real images: {output}. No inference results have been generated.')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/public_ppe_manifest.json')
    parser.add_argument('--media-dir', type=Path, default=ROOT / 'data/public_ppe/images')
    parser.add_argument('--per-class', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    args = parser.parse_args()
    prepare(args.output, args.media_dir, args.per_class, args.seed, exclude_manifests=args.exclude_manifest)


if __name__ == '__main__':
    main()
