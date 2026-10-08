"""Append-only JSONL audit records, using only the Python standard library.

Writers and readers coordinate with POSIX file locks. Invalid chains are never
extended. Verification reports a zero-based line index, even when a record's
claimed sequence number has been altered. An empty log is a valid empty chain.

A hash chain detects edits relative to its recorded history; detecting removal
of the tail or replacement of the entire chain requires an external hash anchor.
"""

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

ZERO_HASH = "0" * 64
_PAYLOAD_FIELDS = {
    "input_name", "input_sha256", "media_type", "model_id", "adapter",
    "prompt_sha256", "label_grounding", "safety_context",
    "raw_output_sha256", "result", "parse_error",
}
_RECORD_FIELDS = _PAYLOAD_FIELDS | {"seq", "timestamp", "prev_hash", "hash"}


def sha256_file(path: str | os.PathLike) -> str:
    """Hash file bytes in bounded chunks without retaining the input."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serialize(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _record_hash(record: dict) -> str:
    return hashlib.sha256(
        _serialize({key: value for key, value in record.items() if key != "hash"})
    ).hexdigest()


def _is_hash(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _is_name(value) -> bool:
    return (
        isinstance(value, str) and value not in {"", ".", ".."}
        and "/" not in value and "\\" not in value
    )


def _valid_record(record) -> bool:
    if not isinstance(record, dict) or record.keys() != _RECORD_FIELDS:
        return False
    if type(record["seq"]) is not int or record["seq"] < 0:
        return False
    try:
        timestamp = datetime.fromisoformat(record["timestamp"])
    except (TypeError, ValueError):
        return False
    if timestamp.utcoffset() != timedelta(0):
        return False
    if not _is_name(record["input_name"]) or record["media_type"] not in ("image", "video"):
        return False
    if not all(_is_hash(record[key]) for key in (
        "input_sha256", "prompt_sha256", "raw_output_sha256", "prev_hash", "hash",
    )):
        return False
    if not isinstance(record["model_id"], str) or not isinstance(record["safety_context"], str):
        return False
    if type(record["label_grounding"]) is not bool or type(record["parse_error"]) is not bool:
        return False
    if record["parse_error"] and record["result"] is not None:
        return False
    adapter = record["adapter"]
    return adapter is None or (
        isinstance(adapter, dict) and adapter.keys() == {"path", "sha256"}
        and _is_name(adapter["path"]) and _is_hash(adapter["sha256"])
    )


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _scan(stream) -> tuple[dict | None, int | None]:
    last = None
    for seq, line in enumerate(stream):
        try:
            record = json.loads(line, object_pairs_hook=_unique_object)
            previous_hash = last["hash"] if last else ZERO_HASH
            if (
                not line.endswith(b"\n") or not _valid_record(record)
                or record["seq"] != seq or record["prev_hash"] != previous_hash
                or record["hash"] != _record_hash(record)
            ):
                return last, seq
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return last, seq
        last = record
    return last, None


def append_record(log_path: str | os.PathLike, **fields) -> dict:
    """Add one record; timestamp, seq, prev_hash and hash are managed here.

    Only the documented payload fields are accepted. Callers supply hashes, not
    media bytes or raw model output. A failed parse must have result=None.
    Corrupt or incomplete existing logs raise ValueError without being changed.
    """
    if fields.keys() != _PAYLOAD_FIELDS:
        raise ValueError("Supply exactly the documented audit payload fields")
    # Snapshot JSON values so changes to the caller's result cannot affect a write.
    record = json.loads(_serialize(fields))
    record.update(seq=0, timestamp=datetime.now(timezone.utc).isoformat(),
                  prev_hash=ZERO_HASH, hash=ZERO_HASH)
    if not _valid_record(record):
        raise ValueError("Invalid audit payload")

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        last, broken_seq = _scan(stream)
        if broken_seq is not None:
            raise ValueError(f"Audit chain is invalid at seq {broken_seq}")
        record.update(
            seq=last["seq"] + 1 if last else 0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prev_hash=last["hash"] if last else ZERO_HASH,
        )
        record["hash"] = _record_hash(record)
        stream.write(_serialize(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def verify_log(log_path: str | os.PathLike) -> tuple[bool, int | None]:
    """Verify the chain; the failure index is the zero-based physical line."""
    try:
        with open(log_path, "rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_SH)
            _, broken_seq = _scan(stream)
    except FileNotFoundError:
        return False, 0
    return broken_seq is None, broken_seq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Verify a JSONL audit chain")
    verify.add_argument("log_path")
    args = parser.parse_args(argv)
    try:
        valid, broken_seq = verify_log(args.log_path)
    except OSError as error:
        print(f"Cannot verify audit log: {error}")
        return 1
    if valid:
        print("Audit log verified: OK")
        return 0
    print(f"Audit log INVALID at seq {broken_seq} (zero-based line)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
