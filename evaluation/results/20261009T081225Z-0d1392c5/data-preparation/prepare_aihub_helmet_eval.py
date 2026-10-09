"""Select paired helmet images/labels from user-supplied AI Hub 71770 archives.

Read ZIP members directly: never unpack archive paths onto the filesystem.
This script does not train a model or reinterpret provider scores as our scores.
"""
import argparse
import hashlib
import json
import os
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r'^S63_DATA3_(H[01])_(L\d+)_D(\d{4}-\d{2}-\d{2})-(\d{2}-\d{2})_(\d+)_(\d+)\.(jpg|json)$')
CLASSES = {'안전모 미착용머리': True, '안전모 착용머리': False}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def member_info(name):
    # AI Hub uses a leading slash in these ZIPs. Accept only a single basename.
    name = name.lstrip('/')
    if len(PurePosixPath(name).parts) != 1 or '\\' in name:
        raise ValueError('Unexpected archive member path')
    match = PATTERN.fullmatch(name)
    if not match:
        raise ValueError('Unexpected helmet source filename')
    kind, location, date, minute, sequence, _frame, extension = match.groups()
    return {'name': name, 'kind': kind, 'group': f'aihub:71770:{location}:{date}',
            'sequence': f'{kind}_{location}_{date}-{minute}_{sequence}', 'extension': extension}


def label_target(label):
    classes = {item['id']: CLASSES[item['name']] for item in label['class']}
    annotations = label.get('annotations', [])
    if not annotations:
        raise ValueError('No annotated heads')
    labels = [classes[item['object_class']] for item in annotations]
    for item, missing in zip(annotations, labels):
        expected = '안전모 미착용' if missing else '안전모 착용'
        values = [entry['value'] for entry in item.get('categories', []) if entry.get('name') == '안전장비']
        if values != [expected]:
            raise ValueError('Class and object category disagree')
    return any(labels), len(annotations)


def archive_path(source, split, kind, labels=False, required=True):
    prefix = ('VL' if labels else 'VS') if split == 'Validation' else ('TL' if labels else 'TS')
    matches = list(source.rglob(f'{prefix}_*S63_DATA3_*{kind}.zip'))
    if not matches and not required:
        return None
    if len(matches) != 1:
        raise ValueError(f'Expected one {prefix} helmet {kind} archive; found {len(matches)}')
    return matches[0]


def prepare(source, output, media_dir, per_class=50, seed=3042):
    if per_class < 1:
        raise ValueError('per_class must be positive')
    rng = random.Random(seed)
    records, inventory, seen_hashes = [], [], set()
    training_groups = set()
    training_archives_checked = []
    for kind in ('H0', 'H1'):
        path = archive_path(source, 'Training', kind, labels=True, required=False)
        if path is None:
            continue
        training_archives_checked.append(path.name)
        with zipfile.ZipFile(path) as archive:
            training_groups.update(member_info(item.filename)['group'] for item in archive.infolist() if not item.is_dir())
    media_dir.mkdir(parents=True, exist_ok=True)
    (media_dir.parent / 'labels').mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded = Counter()
    excluded_examples = []
    for kind in ('H0', 'H1'):
        images_path = archive_path(source, 'Validation', kind)
        labels_path = archive_path(source, 'Validation', kind, labels=True)
        with zipfile.ZipFile(images_path) as images, zipfile.ZipFile(labels_path) as labels:
            media_members = {}
            for item in images.infolist():
                if not item.is_dir():
                    info = member_info(item.filename)
                    if info['extension'] != 'jpg' or info['kind'] != kind or info['name'] in media_members:
                        raise ValueError('Unexpected or duplicate image member')
                    media_members[info['name']] = item
            candidates = defaultdict(list)
            label_names = set()
            for item in labels.infolist():
                if item.is_dir():
                    continue
                info = member_info(item.filename)
                if info['extension'] != 'json' or info['kind'] != kind or info['name'] in label_names:
                    raise ValueError('Unexpected or duplicate label member')
                label_names.add(info['name'])
                if item.file_size > 2 * 1024 * 1024:
                    raise ValueError('Unexpectedly large annotation')
                raw = labels.read(item)
                label = json.loads(raw.decode('utf-8-sig'))
                image_name = info['name'].removesuffix('.json') + '.jpg'
                if label['images']['file_name'] != image_name or image_name not in media_members:
                    raise ValueError('Image/annotation pairing mismatch')
                try:
                    target, count = label_target(label)
                    if target != (kind == 'H0'):
                        raise ValueError('Archive category mismatch')
                except (KeyError, ValueError, TypeError) as exc:
                    excluded['invalid_or_inconsistent_annotations'] += 1
                    if len(excluded_examples) < 20:
                        excluded_examples.append({'archive': labels_path.name, 'member': item.filename,
                                                  'reason': str(exc)})
                    continue
                candidates[info['sequence']].append((info, image_name, raw, label['images'], target, count))
            if len(media_members) != len(label_names):
                raise ValueError('Unpaired image or annotation counts')
            # Round robin across source sequences before selecting additional frames.
            sequences = sorted(candidates)
            rng.shuffle(sequences)
            for pool in candidates.values():
                pool.sort(key=lambda item: item[0]['name'])
                rng.shuffle(pool)
            selected = 0
            while sequences and selected < per_class:
                for sequence in list(sequences):
                    info, image_name, raw_label, metadata, target, count = candidates[sequence].pop()
                    if not candidates[sequence]:
                        sequences.remove(sequence)
                    member = media_members[image_name]
                    if member.file_size > 20 * 1024 * 1024:
                        raise ValueError('Unexpectedly large image')
                    content = images.read(member)  # ZIP CRC verified by zipfile.
                    sha = hashlib.sha256(content).hexdigest()
                    if sha in seen_hashes:
                        excluded['duplicate_media'] += 1
                        continue
                    seen_hashes.add(sha)
                    destination = media_dir / image_name
                    destination.write_bytes(content)
                    (media_dir.parent / 'labels' / info['name']).write_bytes(raw_label)
                    records.append({
                        'id': 'aihub:71770:' + Path(image_name).stem, 'group_id': info['group'],
                        'media_path': os.path.relpath(destination.resolve(), output.parent.resolve()),
                        'media_sha256': sha, 'is_hazardous': target,
                        'events': ['ppe_violation'] if target else [],
                        'annotation_source': 'AI Hub 71770: upstream object class and safety-equipment category',
                        'annotation_sha256': hashlib.sha256(raw_label).hexdigest(),
                        'source_archive': images_path.name, 'source_member': member.filename,
                        'source_sequence': sequence, 'upstream_split': 'Validation',
                        'annotation_objects': count, 'capture': metadata,
                    })
                    selected += 1
                    if selected == per_class:
                        break
            if selected != per_class:
                raise ValueError('Insufficient valid distinct images')
            inventory.append({'kind': kind, 'paired_files': len(media_members), 'selected': selected,
                              'source_sequences': len(candidates),
                              'archives': [{'path': str(p.relative_to(source)), 'sha256': digest(p),
                                            'bytes': p.stat().st_size} for p in (images_path, labels_path)]})
    rng.shuffle(records)
    groups = {record['group_id'] for record in records}
    manifest = {
        'name': 'aihub-smart-yard-helmet-v1', 'source': 'https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71770',
        'acquisition': 'User-supplied local archives; images and labels are not redistributed in Git',
        'preparation_script_sha256': digest(Path(__file__)),
        'seed': seed, 'samples_per_class': per_class,
        'safety_context': 'Inspect safety helmet compliance only; a missing helmet is ppe_violation.',
        'scope': 'Image-level helmet presence only, not site-wide safety or localization accuracy',
        'group_definition': 'Conservative filename location code + capture day across classes and upstream splits',
        'official_training_group_overlap': sorted(groups & training_groups),
        'training_archives_checked': training_archives_checked,
        'selected_groups': len(groups), 'selected_sequences': len({r['source_sequence'] for r in records}),
        'inventory': inventory, 'excluded': dict(excluded), 'excluded_examples': excluded_examples,
        'limitations': [
            'Upstream labels have not been independently reviewed by a safety expert.',
            'Correlated frames from shared sessions are not independent field trials.',
            'Official split names do not establish session independence; group overlap is reported.',
            'No project training uses these archives; pretraining overlap is unknown.',
            'Only annotated helmet compliance is scored; all visible heads may not be annotated.',
            'A shipyard dataset does not establish performance across manufacturing industries.',
        ],
        'samples': records,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'evaluation/aihub_helmet_test_v1.json')
    parser.add_argument('--media-dir', type=Path, default=ROOT / 'data/aihub_helmet_v1/images')
    parser.add_argument('--per-class', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3042)
    args = parser.parse_args()
    result = prepare(args.source_root, args.output, args.media_dir, args.per_class, args.seed)
    print(json.dumps({k: result[k] for k in ('selected_groups', 'selected_sequences', 'official_training_group_overlap', 'excluded')},
                     ensure_ascii=False, indent=2))
    print(f"Prepared {len(result['samples'])} paired images: {args.output}")


if __name__ == '__main__':
    main()
