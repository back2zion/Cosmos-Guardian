"""Reproducible held-out evaluation through the production agent.

python -m scripts.benchmark --manifest evaluation/manifest.json --preflight
python -m scripts.benchmark --manifest evaluation/manifest.json --training-manifest evaluation/training.json
"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Also support `python scripts/benchmark.py` from any working directory.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audit import sha256_file
from core.evaluation import (
    AbstainedPrediction,
    load_manifest,
    prediction,
    summarize,
    validate_holdout,
)


def training_ids():
    ids = set()
    for name in ("fine_tuning_data.json", "fine_tuning_construction.json"):
        for item in json.loads((ROOT / "scripts" / name).read_text()):
            ids.add(str(item["id"]))
    return ids


def preflight(args):
    errors, samples, training = [], [], []
    context, scope, limitations = "Industrial safety assessment", "Industrial safety", []
    try:
        samples = load_manifest(args.manifest)
        metadata = json.loads(args.manifest.read_text(encoding="utf-8"))
        context = metadata.get("safety_context", context)
        scope = metadata.get("scope", scope)
        limitations = metadata.get("limitations", limitations)
        if args.training_manifest:
            training = load_manifest(args.training_manifest)
        elif not args.base_only:
            errors.append("A training manifest is required for LoRA comparison (IDs, source groups and media hashes)")
        validate_holdout(samples, training_ids(), training)
        if getattr(args, 'development_manifest', None):
            development = load_manifest(args.development_manifest)
            if ({s['id'] for s in samples} & {s['id'] for s in development}
                    or {s['media_sha256'] for s in samples} & {s['media_sha256'] for s in development}):
                errors.append('Development/evaluation ID or content overlap')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    if not args.base_only:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (args.adapter / name).is_file():
                errors.append(f"Missing LoRA artifact: {args.adapter / name}")
    dependencies = ["torch", "transformers", "yaml", "PIL", "cv2"] + ([] if args.base_only else ["peft"])
    for name in dependencies:
        if importlib.util.find_spec(name) is None:
            errors.append(f"Missing inference dependency: {name}")
    return {"ready": not errors, "errors": errors, "samples": len(samples),
            "training_samples": len(training), "mode": "base-only" if args.base_only else "base-vs-lora",
            "group_and_hash_holdout_checked": bool(training),
            "safety_context": context, "scope": scope, "dataset_limitations": limitations,
            "manifest": str(args.manifest.resolve()), "adapter": str(args.adapter.resolve())}, samples


def evaluate(agent, samples, safety_context="Industrial safety assessment", on_record=None):
    records = []
    for sample in samples:
        start = time.perf_counter()
        record = {**sample}
        result = None
        try:
            for update in agent.analyze_media(sample["media_path"], safety_context=safety_context):
                if update["stage"] == "complete":
                    result = update["result"]
            detected, events = prediction(result)
            record.update(hazard_detected=detected, predicted_events=events, result=result,
                          latency_seconds=round(time.perf_counter() - start, 3))
        except Exception as exc:  # noqa: BLE001 - account for each failed sample without inventing predictions
            record.update(error=f"{type(exc).__name__}: {exc}", elapsed_seconds=round(time.perf_counter() - start, 3))
            if result is not None:
                record['result'] = result
            if isinstance(exc, AbstainedPrediction):
                record['abstained'] = True
        records.append(record)
        if on_record is not None:
            on_record(record)
        print(f"{sample['id']}: {'ERROR' if record.get('error') else 'OK'}", flush=True)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation/manifest.json")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument('--development-manifest', type=Path, help='Check ID/hash disjointness; source session independence is not implied')
    parser.add_argument('--profile', choices=['legacy', 'helmet'], default='legacy')
    parser.add_argument("--adapter", type=Path, default=ROOT / "outputs/cosmos-reason2-2b-safety-lora")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/benchmark_results")
    parser.add_argument("--base-only", action="store_true", help="Explicitly evaluate only the base model; never label it LoRA")
    parser.add_argument("--preflight", action="store_true", help="Validate inputs without loading models")
    args = parser.parse_args(argv)
    if args.profile == 'helmet' and not args.base_only:
        parser.error('Helmet profile currently supports --base-only')
    check, samples = preflight(args)
    if args.preflight:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "preflight.json").write_text(
            json.dumps({"status": "preflight_only", "timestamp": datetime.now(timezone.utc).isoformat(), **check},
                       ensure_ascii=False, indent=2), encoding="utf-8",
        )
    print(json.dumps(check, ensure_ascii=False, indent=2))
    if not check["ready"]:
        return 2
    if args.preflight:
        return 0

    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "data/huggingface"))
    import torch
    import transformers

    from core.agent import CosmosGuardianAgent

    agent_class = CosmosGuardianAgent
    source_paths = ['core/agent.py', 'core/evaluation.py', 'scripts/benchmark.py', 'prompts/safety_inspector.yaml',
                    'pyproject.toml', 'uv.lock']
    prompt_path = ROOT / 'prompts/safety_inspector.yaml'
    if args.profile == 'helmet':
        from core.helmet_agent import HelmetAgent
        agent_class = HelmetAgent
        prompt_path = ROOT / 'prompts/helmet_inspector.yaml'
        source_paths += ['core/helmet.py', 'core/helmet_agent.py', 'prompts/helmet_inspector.yaml']

    if not torch.cuda.is_available():
        print("CUDA is unavailable; no benchmark was run", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"{stamp}-{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    report = {"status": "running", "setup": {
        **check, "seed": 0, "timestamp": stamp, "git_revision": revision.stdout.strip(),
        "working_tree_dirty": bool(dirty.stdout.strip()), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0), "manifest_sha256": sha256_file(args.manifest),
        "prompt_sha256": sha256_file(prompt_path), 'profile': args.profile,
        'development_manifest_sha256': sha256_file(args.development_manifest) if args.development_manifest else None,
        "agent_code_sha256": sha256_file(ROOT / "core/agent.py"),
        "evaluation_code_sha256": sha256_file(ROOT / "core/evaluation.py"),
        "benchmark_code_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": None if args.base_only else sha256_file(args.adapter / "adapter_model.safetensors"),
        "training_manifest_sha256": sha256_file(args.training_manifest) if args.training_manifest else None,
        "label_grounding": False, "classification": "exact event_type categories; unclassified outputs count as mismatches",
    }, "summary": {}, "details": {}}
    report['setup']['source_sha256'] = {name: sha256_file(ROOT / name) for name in source_paths}
    for name in source_paths:
        destination = run_dir / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(args.manifest, run_dir / 'manifest.json')
    output = run_dir / "benchmark_results.json"

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(output)

    save()
    try:
        for mode in (["base"] if args.base_only else ["base", "lora"]):
            agent = agent_class(use_label_grounding=False)
            agent.load_model(adapter_path=str(args.adapter) if mode == "lora" else None)
            if mode == "lora" and not agent.adapter_loaded:
                raise RuntimeError("LoRA loading failed; refusing to publish base results as LoRA")
            report["setup"][f"{mode}_model_id"] = agent.model_id
            report["setup"][f"{mode}_model_revision"] = getattr(agent.model.config, "_commit_hash", None)
            report["details"][mode] = []

            def checkpoint(record, mode=mode):
                report["details"][mode].append(record)
                report["progress"] = {"mode": mode, "completed_samples": len(report["details"][mode]), "total_samples": len(samples)}
                save()

            records = evaluate(agent, samples, check["safety_context"], on_record=checkpoint)
            report["details"][mode] = records
            report["summary"][mode] = summarize(records)
            save()
            del agent
            torch.cuda.empty_cache()
        report["status"] = "completed_with_errors" if any(r.get("error") for rows in report["details"].values() for r in rows) else "completed"
    except Exception as exc:  # noqa: BLE001 - retain a clearly failed artifact if model loading fails
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    save()
    print(f"Results: {output}")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
