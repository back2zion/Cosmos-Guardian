"""Benchmark validation and metrics without importing inference dependencies."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .audit import sha256_file

EVENTS = {"entrapment", "fall", "struck_by", "ppe_violation", "other"}


class AbstainedPrediction(ValueError):
    """An explicit unknown is neither a safe prediction nor a schema failure."""


def load_manifest(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("samples"), list) or not data["samples"]:
        raise ValueError("Manifest must contain a nonempty samples array")
    if "safety_context" in data and (not isinstance(data["safety_context"], str)
                                     or not data["safety_context"].strip() or len(data["safety_context"]) > 4000):
        raise ValueError("Manifest safety_context must be nonempty text of at most 4000 characters")
    seen_ids, seen_hashes, samples = set(), set(), []
    for item in data["samples"]:
        if not isinstance(item, dict):
            raise TypeError("Each sample must be an object")
        for key in ("id", "group_id", "media_path", "annotation_source"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"Each sample requires a nonempty {key}")
        if item["id"] in seen_ids:
            raise ValueError(f"Duplicate sample ID: {item['id']}")
        seen_ids.add(item["id"])
        if type(item.get("is_hazardous")) is not bool:
            raise ValueError(f"{item['id']}: is_hazardous must be an explicit boolean")
        events = item.get("events")
        if not isinstance(events, list) or not all(isinstance(e, str) and e in EVENTS for e in events):
            raise ValueError(f"{item['id']}: unsupported event labels")
        if bool(events) != item["is_hazardous"]:
            raise ValueError(f"{item['id']}: events must agree with is_hazardous")
        media = (path.parent / item["media_path"]).resolve()
        if not media.is_file():
            raise ValueError(f"Missing media: {media}")
        digest = sha256_file(media)
        if "media_sha256" in item and item["media_sha256"] != digest:
            raise ValueError(f"Media hash mismatch: {item['id']}")
        if digest in seen_hashes:
            raise ValueError(f"Duplicate media content: {item['id']}")
        seen_hashes.add(digest)
        samples.append({**item, "media_path": str(media), "media_sha256": digest})
    return samples


def validate_holdout(samples, training_ids, training_samples=()):
    train_ids = set(training_ids) | {s["id"] for s in training_samples}
    train_groups = {s["group_id"] for s in training_samples}
    train_hashes = {s["media_sha256"] for s in training_samples}
    for sample in samples:
        if (sample["id"] in train_ids or Path(sample["media_path"]).stem in train_ids
                or sample["group_id"] in train_groups or sample["media_sha256"] in train_hashes):
            raise ValueError(f"Training/evaluation overlap: {sample['id']}")
    if {s["is_hazardous"] for s in samples} != {True, False}:
        raise ValueError("Evaluation requires both hazardous and safe examples to measure false alarms")


def prediction(result):
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError("Invalid model JSON")
    if result.get('assessment_status') == 'unknown':
        raise AbstainedPrediction('Model abstained: human inspection required')
    hazards = result.get("hazards_detected")
    if not isinstance(hazards, list):
        raise TypeError("Missing hazard list")
    categories = set()
    for hazard in hazards:
        if not isinstance(hazard, dict) or not isinstance(hazard.get("type"), str) or not hazard["type"].strip():
            raise ValueError("Invalid hazard entry")
        category = hazard.get("event_type")
        categories.add(category if isinstance(category, str) and category in EVENTS else "unclassified")
    return bool(hazards), sorted(categories)


def summarize(records):
    """Failures remain unknown; never treat an error as a safe prediction."""
    matrix = Counter(tp=0, fp=0, tn=0, fn=0, invalid_hazardous=0, invalid_safe=0)
    event_correct, latencies = 0, []
    per_event = {event: Counter(tp=0, fp=0, fn=0) for event in sorted(EVENTS)}
    for record in records:
        truth = record["is_hazardous"]
        if record.get("error"):
            matrix["invalid_hazardous" if truth else "invalid_safe"] += 1
            continue
        detected = record["hazard_detected"]
        matrix[("tp" if detected else "fn") if truth else ("fp" if detected else "tn")] += 1
        target, predicted = set(record["events"]), set(record["predicted_events"])
        event_correct += target == predicted
        for event, counts in per_event.items():
            if event in target:
                counts["tp" if event in predicted else "fn"] += 1
            elif event in predicted:
                counts["fp"] += 1
        latencies.append(record["latency_seconds"])

    def ratio(n, d):
        return round(n / d, 6) if d else None

    positives = sum(r["is_hazardous"] for r in records)
    negatives = len(records) - positives
    ordered = sorted(latencies)
    return {
        "samples": len(records), "hazardous_samples": positives, "safe_samples": negatives,
        "abstentions": sum(bool(r.get('abstained')) for r in records),
        "schema_error_count": sum(bool(r.get('error')) and not r.get('abstained', False) for r in records),
        "confusion": dict(matrix), "valid_output_rate": ratio(len(latencies), len(records)),
        "precision": ratio(matrix["tp"], matrix["tp"] + matrix["fp"]),
        "recall_valid_outputs": ratio(matrix["tp"], matrix["tp"] + matrix["fn"]),
        "effective_recall_including_failures": ratio(matrix["tp"], positives),
        "false_positive_rate_valid_outputs": ratio(matrix["fp"], matrix["fp"] + matrix["tn"]),
        "event_exact_match_rate": ratio(event_correct, len(records)),
        "per_event_valid_outputs": {e: dict(c) for e, c in per_event.items()},
        "mean_latency_seconds": round(mean(latencies), 3) if latencies else None,
        "p95_latency_seconds": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)] if ordered else None,
        "latency_scope": "successful analysis only, full pipeline excluding model loading",
    }
