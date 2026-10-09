import json
import zipfile

import pytest

from scripts.prepare_aihub_helmet_eval import label_target, member_info, prepare


def make_label(name, missing):
    category = '안전모 미착용' if missing else '안전모 착용'
    return {'class': [{'id': 0, 'name': category + '머리'}], 'images': {'file_name': name},
            'annotations': [{'object_class': 0, 'categories': [{'name': '안전장비', 'value': category}]}]}


def test_source_groups_cross_class_and_official_split_names():
    positive = member_info('/S63_DATA3_H0_L1_D2023-10-25-15-49_021_004498.jpg')
    negative = member_info('S63_DATA3_H1_L1_D2023-10-25-13-40_001_003646.json')
    assert positive['group'] == negative['group']
    for name in ['../../escape.jpg', '/other/file.jpg', 'C:\\escape.jpg']:
        with pytest.raises(ValueError):
            member_info(name)


def test_labels_not_folder_names_determine_targets():
    assert label_target(make_label('a.jpg', True)) == (True, 1)
    assert label_target(make_label('b.jpg', False)) == (False, 1)
    malformed = make_label('a.jpg', True)
    malformed['annotations'][0]['categories'][0]['value'] = '안전모 착용'
    with pytest.raises(ValueError, match='disagree'):
        label_target(malformed)


def test_zip_pairing_hashes_and_recorded_group_overlap(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for kind, missing in [('H0', True), ('H1', False)]:
        name = f'S63_DATA3_{kind}_L1_D2023-10-25-13-40_001_000001.jpg'
        with zipfile.ZipFile(source / f'VS_S63_DATA3_{kind}.zip', 'w') as media:
            media.writestr('/' + name, kind.encode())
        with zipfile.ZipFile(source / f'VL_S63_DATA3_{kind}.zip', 'w') as labels:
            labels.writestr('/' + name.replace('.jpg', '.json'), json.dumps(make_label(name, missing)))
        with zipfile.ZipFile(source / f'TL_S63_DATA3_{kind}.zip', 'w') as training:
            training.writestr('/' + name.replace('.jpg', '.json'), '{}')
    manifest = prepare(source, tmp_path / 'manifest.json', tmp_path / 'media/images', per_class=1)
    assert len(manifest['samples']) == 2
    assert manifest['selected_groups'] == 1
    assert manifest['official_training_group_overlap'] == ['aihub:71770:L1:2023-10-25']
    assert {item['is_hazardous'] for item in manifest['samples']} == {True, False}
    assert len(list((tmp_path / 'media/labels').glob('*.json'))) == 2
