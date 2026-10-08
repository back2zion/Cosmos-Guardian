import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.audit import append_record, sha256_file, verify_log

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fields():
    return {
        "input_name": "작업장.png",
        "input_sha256": hashlib.sha256(b"image bytes").hexdigest(),
        "media_type": "image",
        "model_id": "nvidia/Cosmos-Reason2-2B",
        "adapter": None,
        "prompt_sha256": hashlib.sha256(b"systemuser").hexdigest(),
        "label_grounding": False,
        "safety_context": "산업 안전",
        "raw_output_sha256": hashlib.sha256(b"model output").hexdigest(),
        "result": {"hazards_detected": [], "overall_safety_score": 90},
        "parse_error": False,
    }


@pytest.fixture
def chain(tmp_path, fields):
    path = tmp_path / "audit" / "decisions.jsonl"
    records = [append_record(path, **fields) for _ in range(3)]
    return path, records


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_three_records_and_canonical_hashes(chain):
    path, records = chain
    assert verify_log(path) == (True, None)
    assert records[0]["prev_hash"] == "0" * 64
    assert [record["seq"] for record in records] == [0, 1, 2]
    previous = "0" * 64
    for record in records:
        assert record["prev_hash"] == previous
        payload = {key: value for key, value in record.items() if key != "hash"}
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        assert record["hash"] == hashlib.sha256(encoded).hexdigest()
        assert datetime.fromisoformat(record["timestamp"]).utcoffset() == timedelta(0)
        previous = record["hash"]
    assert "작업장" in path.read_text(encoding="utf-8")


def test_appending_preserves_existing_bytes(chain, fields):
    path, _ = chain
    original = path.read_bytes()
    record = append_record(path, **fields)
    assert path.read_bytes().startswith(original)
    assert record["seq"] == 3
    assert verify_log(path) == (True, None)


def test_middle_result_tampering(chain):
    path, records = chain
    records[1]["result"]["overall_safety_score"] = 0
    write_records(path, records)
    assert verify_log(path) == (False, 1)


def test_deleted_middle_line(chain):
    path, records = chain
    write_records(path, [records[0], records[2]])
    assert verify_log(path) == (False, 1)


def test_forged_record_appended_without_rehash(chain):
    path, records = chain
    forged = {**records[-1], "seq": 3, "prev_hash": records[-1]["hash"]}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(forged) + "\n")
    assert verify_log(path) == (False, 3)


def test_wrong_sequence_even_if_rehashed(chain):
    path, records = chain
    records[1]["seq"] = 900
    payload = {key: value for key, value in records[1].items() if key != "hash"}
    records[1]["hash"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()
    write_records(path, records)
    assert verify_log(path) == (False, 1)


@pytest.mark.parametrize("bad_line", [b"{bad json}\n", b"\xff\n", b"{}\n", b"[]\n", b"\n"])
def test_invalid_lines_report_physical_index(chain, bad_line):
    path, _ = chain
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + bad_line + lines[2])
    assert verify_log(path) == (False, 1)


def test_truncated_last_line_cannot_be_extended(chain, fields):
    path, _ = chain
    incomplete = path.read_bytes().removesuffix(b"\n")
    path.write_bytes(incomplete)
    assert verify_log(path) == (False, 2)
    with pytest.raises(ValueError, match="seq 2"):
        append_record(path, **fields)
    assert path.read_bytes() == incomplete


def test_corrupt_chain_cannot_be_extended(chain, fields):
    path, records = chain
    records[0]["result"] = {"tampered": True}
    write_records(path, records)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="seq 0"):
        append_record(path, **fields)
    assert path.read_bytes() == original


def test_empty_and_missing_log(tmp_path):
    path = tmp_path / "empty.jsonl"
    assert verify_log(path) == (False, 0)
    path.touch()
    assert verify_log(path) == (True, None)


@pytest.mark.parametrize("changes", [
    {"raw_output": "private model output"},
    {"seq": 100},
    {"input_name": "/private/frame.png"},
    {"input_name": "C:\\private\\frame.png"},
    {"media_type": "audio"},
    {"input_sha256": "invalid"},
    {"parse_error": True},
    {"adapter": {"path": "/private/adapter", "sha256": "a" * 64}},
    {"result": {"score": float("nan")}},
])
def test_invalid_payload_is_not_written(tmp_path, fields, changes):
    path = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError):
        append_record(path, **(fields | changes))
    assert not path.exists()


def test_parse_failure_and_adapter_metadata(tmp_path, fields):
    fields.update(
        adapter={"path": "safety-lora", "sha256": "a" * 64},
        parse_error=True, result=None,
    )
    path = tmp_path / "decisions.jsonl"
    record = append_record(path, **fields)
    assert record["result"] is None
    assert record["parse_error"] is True
    assert record["adapter"] == fields["adapter"]
    assert verify_log(path) == (True, None)
    assert os.stat(path).st_mode & 0o777 == 0o600


def append_from_worker(args):
    path, fields = args
    return append_record(path, **fields)


@pytest.mark.parametrize("executor_type", [ThreadPoolExecutor, ProcessPoolExecutor])
def test_concurrent_writers(tmp_path, fields, executor_type):
    path = tmp_path / "concurrent.jsonl"
    with executor_type(max_workers=4) as executor:
        records = list(executor.map(append_from_worker, [(path, fields)] * 12))
    assert sorted(record["seq"] for record in records) == list(range(12))
    assert verify_log(path) == (True, None)


def test_file_hash_uses_all_bytes(tmp_path):
    data = b"\x00\xffprivate input" * 100_000
    path = tmp_path / "media.bin"
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_cli_exit_codes(chain):
    path, records = chain
    command = [sys.executable, "-S", "-m", "core.audit", "verify", str(path)]
    valid = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert valid.returncode == 0
    assert "OK" in valid.stdout
    records[1]["result"] = None
    write_records(path, records)
    invalid = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    assert invalid.returncode == 1
    assert "seq 1" in invalid.stdout


def test_import_needs_only_standard_library():
    script = (
        "import sys; import core.audit; "
        "assert not {'torch', 'transformers', 'peft'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
