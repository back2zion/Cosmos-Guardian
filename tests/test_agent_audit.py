"""Exercise the agent's real audit flow with lightweight inference stubs."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from core.audit import verify_log


@pytest.fixture
def agent_runtime(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: False, is_bf16_supported=lambda: False,
        empty_cache=lambda: None,
    )
    torch.float16 = "float16"
    torch.bfloat16 = "bfloat16"
    transformers = ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "yaml", ModuleType("yaml"))

    spec = importlib.util.spec_from_file_location(
        "core._agent_under_test", Path(__file__).resolve().parents[1] / "core" / "agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.CosmosGuardianAgent, "_load_prompts", lambda self: {
        "system_instructions": "안전 분석 system\n",
        "user_prompt_template": "Inspect {safety_context}.",
    })

    class Inputs(dict):
        input_ids = SimpleNamespace(shape=(1, 2))

        def to(self, device):
            return self

    class Processor:
        tokenizer = object()
        image_processor = SimpleNamespace()

        def apply_chat_template(self, conversation, **kwargs):
            self.conversation = conversation
            return Inputs()

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            return None

    def make_agent(output='{"overall_safety_score": 80}', **kwargs):
        processor = Processor()
        model = Model()
        transformers.TextIteratorStreamer = lambda *args, **kwargs: iter([output])
        transformers.AutoModelForImageTextToText = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model,
        )
        transformers.AutoProcessor = SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: processor,
        )
        agent = module.CosmosGuardianAgent(**kwargs)
        agent.model = model
        agent.processor = processor
        return agent

    return SimpleNamespace(module=module, make_agent=make_agent)


def test_disabled_audit_preserves_complete_and_does_not_hash(agent_runtime, monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Disabled audit must not perform audit I/O")

    monkeypatch.setattr(agent_runtime.module, "sha256_file", unexpected_call)
    monkeypatch.setattr(agent_runtime.module, "append_record", unexpected_call)
    agent = agent_runtime.make_agent()
    events = list(agent.analyze_media("unused.png"))
    assert events[-1] == {"stage": "complete", "result": {"overall_safety_score": 80}}


@pytest.mark.parametrize("grounding", [False, True])
def test_audit_tracks_actual_input_prompts_and_output(agent_runtime, tmp_path, grounding):
    media = tmp_path / "camera.png"
    media.write_bytes(b"PRIVATE_MEDIA_BYTES")
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "camera.json").write_text(json.dumps({
        "meta_information": {"environment": "factory", "event": "jam"},
        "object_information_facility": {"facility_name": "press"},
    }), encoding="utf-8")
    log = tmp_path / "audit.jsonl"
    raw_output = 'PRIVATE_OUTPUT_PREFIX\n{"overall_safety_score": 80}'
    agent = agent_runtime.make_agent(
        raw_output, audit_log_path=str(log), use_label_grounding=grounding,
        label_dir=str(labels),
    )
    events = list(agent.analyze_media(str(media), safety_context="기계 안전"))
    record = json.loads(log.read_text(encoding="utf-8"))
    conversation = agent.processor.conversation
    actual_system = conversation[0]["content"][0]["text"]
    actual_user = conversation[1]["content"][-1]["text"]
    assert record["prompt_sha256"] == hashlib.sha256(
        (actual_system + actual_user).encode("utf-8")
    ).hexdigest()
    assert ("CRITICAL CONTEXT" in actual_system) is grounding
    assert record["label_grounding"] is grounding
    assert record["input_name"] == "camera.png"
    assert record["input_sha256"] == hashlib.sha256(media.read_bytes()).hexdigest()
    assert record["raw_output_sha256"] == hashlib.sha256(raw_output.encode()).hexdigest()
    assert record["safety_context"] == "기계 안전"
    assert record["model_id"] == agent.model_id
    assert record["media_type"] == "image"
    assert record["adapter"] is None
    assert record["result"] == events[-1]["result"]
    assert record["parse_error"] is False
    assert events[-1]["audit"] == {"seq": 0, "hash": record["hash"]}
    assert verify_log(log) == (True, None)
    stored = log.read_bytes()
    assert b"PRIVATE_MEDIA_BYTES" not in stored
    assert b"PRIVATE_OUTPUT_PREFIX" not in stored
    assert str(tmp_path).encode() not in stored


@pytest.mark.parametrize("raw_output", ["PRIVATE_UNPARSEABLE_OUTPUT", "{broken JSON PRIVATE}"])
def test_parse_failures_never_store_raw_output(agent_runtime, tmp_path, raw_output):
    media = tmp_path / "camera.png"
    media.write_bytes(b"image")
    log = tmp_path / "audit.jsonl"
    agent = agent_runtime.make_agent(raw_output, audit_log_path=str(log))
    complete = list(agent.analyze_media(str(media)))[-1]
    record = json.loads(log.read_text())
    assert complete["result"]["raw_output"] == raw_output
    assert record["result"] is None
    assert record["parse_error"] is True
    assert record["raw_output_sha256"] == hashlib.sha256(raw_output.encode()).hexdigest()
    assert raw_output not in log.read_text()


@pytest.mark.parametrize("raw_output, expected", [
    ('{"score": 80', {"score": 80}),
    ('{"score": 80} trailing text', {"score": 80}),
    ('{"error": "reported by model"}', {"error": "reported by model"}),
])
def test_successful_parsing_and_repairs_keep_the_result(agent_runtime, tmp_path, raw_output, expected):
    media = tmp_path / "camera.png"
    media.write_bytes(b"image")
    log = tmp_path / "audit.jsonl"
    agent = agent_runtime.make_agent(raw_output, audit_log_path=str(log))
    complete = list(agent.analyze_media(str(media)))[-1]
    record = json.loads(log.read_text())
    assert record["result"] == complete["result"] == expected
    assert record["parse_error"] is False


def test_audit_failure_does_not_emit_complete(agent_runtime, tmp_path):
    media = tmp_path / "camera.png"
    media.write_bytes(b"image")
    log = tmp_path / "broken.jsonl"
    log.write_text("invalid log\n")
    agent = agent_runtime.make_agent(audit_log_path=str(log))
    events = []
    with pytest.raises(ValueError, match="seq 0"):
        events.extend(agent.analyze_media(str(media)))
    assert all(event["stage"] != "complete" for event in events)
    assert log.read_text() == "invalid log\n"


@pytest.mark.parametrize("load_succeeds", [True, False])
def test_adapter_fingerprint_describes_loaded_weights(agent_runtime, tmp_path, monkeypatch, load_succeeds):
    adapter = tmp_path / "safety-lora"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"weights actually loaded")
    peft = ModuleType("peft")

    def load_adapter(model, path):
        if not load_succeeds:
            raise ValueError("invalid adapter")
        return model

    peft.PeftModel = SimpleNamespace(from_pretrained=load_adapter)
    monkeypatch.setitem(sys.modules, "peft", peft)
    log = tmp_path / "audit.jsonl"
    agent = agent_runtime.make_agent(audit_log_path=str(log))
    agent.load_model(adapter_path=str(adapter))
    # Replacing the file after loading must not relabel the in-memory model.
    weights.write_bytes(b"different weights")
    media = tmp_path / "camera.png"
    media.write_bytes(b"image")
    list(agent.analyze_media(str(media)))
    record = json.loads(log.read_text())
    expected = {
        "path": "safety-lora",
        "sha256": hashlib.sha256(b"weights actually loaded").hexdigest(),
    } if load_succeeds else None
    assert record["adapter"] == expected

    agent.load_model()
    list(agent.analyze_media(str(media)))
    latest = json.loads(log.read_text().splitlines()[-1])
    assert latest["adapter"] is None
