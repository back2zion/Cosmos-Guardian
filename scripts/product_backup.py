"""Back up the product database and referenced original images; restore into a new directory.

Secrets/model weights are not included. Stop API/worker before switching to a restore.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def backup(database, uploads, destination):
    database, uploads, destination = Path(database), Path(uploads), Path(destination)
    if not database.is_file():
        raise ValueError('Database does not exist')
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / 'workflow.sqlite3'
    source = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    snapshot = sqlite3.connect(target)
    try:
        source.backup(snapshot)
        refs = {}
        tables = {r[0] for r in snapshot.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column in [('assessments', 'document'), ('jobs', 'body')]:
            if table not in tables:
                continue
            for row in snapshot.execute(f'SELECT {column} FROM {table}'):  # fixed identifiers only
                item = json.loads(row[0]).get('source', {})
                name = item.get('media_file')
                if name:
                    if Path(name).name != name or name in {'.', '..'}:
                        raise ValueError('Invalid source filename')
                    expected = item['input_sha256']
                    if name in refs and refs[name] != expected:
                        raise ValueError('Conflicting source hashes')
                    refs[name] = expected
    finally:
        source.close()
        snapshot.close()
    (destination / 'uploads').mkdir()
    for name, expected in refs.items():
        original = uploads / name
        if original.is_symlink() or digest(original) != expected:
            raise ValueError('Source image hash mismatch')
        copied = destination / 'uploads' / name
        shutil.copyfile(original, copied)
        if digest(copied) != expected:
            raise ValueError('Copied image hash mismatch')
    files = {'workflow.sqlite3': digest(target), **{'uploads/' + name: value for name, value in refs.items()}}
    manifest = {'format': 'cosmos-product-backup-v1', 'files': files,
                'excludes': ['account secrets', 'model weights', 'legacy JSONL audit log']}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def restore(archive, destination):
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise ValueError('Restore destination must not exist; existing deployments are never overwritten')
    manifest = json.loads((archive / 'manifest.json').read_text())
    if manifest.get('format') != 'cosmos-product-backup-v1' or 'workflow.sqlite3' not in manifest['files']:
        raise ValueError('Unsupported or incomplete backup')
    for name, expected in manifest['files'].items():
        path = Path(name)
        valid = name == 'workflow.sqlite3' or (len(path.parts) == 2 and path.parts[0] == 'uploads' and path.name not in {'.', '..'})
        source = archive / path
        if (not valid or path.is_absolute() or source.is_symlink()
                or not source.resolve().is_relative_to(archive.resolve()) or digest(source) != expected):
            raise ValueError('Invalid backup path or checksum')
    destination.mkdir(parents=True)
    (destination / 'uploads').mkdir()
    for name, expected in manifest['files'].items():
        shutil.copyfile(archive / name, destination / name)
        if digest(destination / name) != expected:
            raise ValueError('Restored file checksum mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('backup')
    create.add_argument('--database', type=Path, default=Path('data/workflow.sqlite3'))
    create.add_argument('--uploads', type=Path, default=Path('uploads'))
    create.add_argument('--destination', required=True, type=Path)
    recover = commands.add_parser('restore')
    recover.add_argument('--archive', required=True, type=Path)
    recover.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    result = backup(args.database, args.uploads, args.destination) if args.command == 'backup' else restore(args.archive, args.destination)
    print(f"Verified {len(result['files'])} files: {args.destination}")


if __name__ == '__main__':
    main()
