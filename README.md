# Cosmos Guardian

A prototype safety-reasoning agent for industrial sites, built on [NVIDIA Cosmos Reason 2](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
Given a CCTV image or short video, it returns detected hazards as JSON: type, severity, a physics-based explanation, a short-horizon prediction, an overall safety score, and a recommended countermeasure.

> **Status: research prototype.** It is not a certified safety system and must not be used to make safety decisions on its own.

[한국어 안내](#한국어-안내)

## What it does

```
image / video ──► video: 6 sampled frames at 336×336
               ──► Cosmos-Reason2-2B (+ optional LoRA adapter)
               ──► JSON report: hazards_detected[], overall_safety_score, recommendation
```

- `core/agent.py`: the agent. It loads the base model, optionally applies a LoRA adapter, and streams progress and reasoning.
- `server.py`: FastAPI backend that streams results over Server-Sent Events (`POST /analyze`, `GET /health`, port 8888).
- `dashboard/`: React/Vite front end for the API.
- `streamlit_app.py`: a single-file Streamlit demo.
- `app.py`: command-line entry point.
- `utils/live_feed.py`: analyzes short clips from a webcam or RTSP stream at fixed intervals.
- `prompts/safety_inspector.yaml`: system prompt and JSON output schema.
- `scripts/`: data conversion, LoRA fine-tuning, and benchmarking.

## Fine-tuning data

Both training sets are built from [AI Hub](https://aihub.or.kr) (Korea) datasets. Only the converted annotation files are in this repository. Images and videos are not.

| File | Domain | Samples | Labels |
|---|---|---|---|
| `scripts/fine_tuning_data.json` | Smart-factory CCTV video | 105 | all `jam` (entrapment) incidents |
| `scripts/fine_tuning_construction.json` | Construction-site images | 1,000 | 5 accident types × 200; 500 hazardous, 500 safe |

The target responses are generated from the labels with templates. They teach the output format and vocabulary, not expert-verified reasoning.

## Benchmark

`scripts/benchmark.py` compares the base model (zero-shot) with the LoRA-adapted model on labeled AI Hub samples. It reports:

- Hazard detection rate
- Event classification accuracy (synonym match against the labeled event)
- Valid safety-score rate
- Latency
- Reasoning keyword score: a rough proxy based on text length and physics keywords. It does not check whether the reasoning is correct.

By default, samples whose IDs appear in the fine-tuning files are excluded. Pass `--include-train` to include them; results will then be inflated.

Benchmark results are not published here yet.

## Run

Requires Python 3.10+ and an NVIDIA GPU. Training configurations for RTX 3090 and H100 are in `scripts/`. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python app.py --media assets/test_data/worker_safety_sample.png --context "Worker safety"
uv run python server.py                       # API on :8888
cd dashboard && npm install && npm run dev    # front end
uv run streamlit run streamlit_app.py         # alternative demo
```

The agent looks for a LoRA adapter at `outputs/cosmos-reason2-2b-safety-lora`. The adapter is not included. Without it, the base model runs zero-shot. Train one with `scripts/train_cosmos_reason2.py` (smart factory) or `scripts/train_construction.py` (construction).

To download AI Hub data with `scripts/download_aihub_real.py`, set `AIHUB_ACCESS_KEY` in your environment. You need your own AI Hub account and approval for each dataset.

## Design notes and limitations

- **Label grounding is off by default.** `CosmosGuardianAgent(use_label_grounding=True)` looks up the AI Hub label for the input file and injects it into the prompt. This is for demos only. Outputs produced this way must not be used to judge accuracy.
- **Small model, small data.** The 2B model and template-generated targets limit how specific the reasoning can be. The smart-factory set covers a single incident type.
- **Output parsing is lenient.** The parser repairs truncated JSON. A repaired report can silently drop fields.
- **Optional audit trail.** Set `COSMOS_AUDIT_LOG=audit/decisions.jsonl` for the API,
  or pass `audit_log_path="audit/decisions.jsonl"` to `CosmosGuardianAgent`.
  Each completed analysis appends a hash-chained JSON record and includes its
  sequence number and hash in the `complete` event. Logs contain the parsed result,
  context, model ID, adapter folder name and hashes of the input, adapter weights,
  actual system/user prompt text (concatenated without a separator), and raw output.
  Media bytes and raw model output are not stored; parse failures record a null result.
  Logging is disabled by default. An audit write failure prevents a `complete` event.
  Verify with `python -m core.audit verify audit/decisions.jsonl` (exit status 0 or 1;
  failure locations are zero-based line indices). POSIX file locks serialize writes,
  and corrupt or incomplete logs cannot be extended. Verification cannot detect
  deletion of the tail or a fully rewritten chain without an externally retained hash.

## Development checks without model dependencies

The audit module uses only the standard library. The tests use stub inference
components, so neither torch nor a GPU is required:

```bash
uv sync --only-dev
uv run --only-dev pytest -q
uv run --only-dev ruff check core tests
```

## License

Code: Apache-2.0. Model weights are subject to NVIDIA's license for Cosmos Reason 2 (see the model card). AI Hub data is subject to AI Hub's terms of use.

---

## 한국어 안내

산업 현장 CCTV 이미지·영상을 받아 위험 요소를 찾고, 물리 기반 설명과 함께 JSON으로 돌려주는 안전 추론 에이전트 프로토타입입니다. NVIDIA Cosmos Reason 2(2B)에 AI Hub 데이터로 LoRA 미세조정을 적용했습니다.

- 인증된 안전 시스템이 아니며, 단독으로 안전 판단에 쓰면 안 됩니다.
- 학습 데이터: 스마트팩토리 끼임 사고 영상 105건, 건설 현장 이미지 1,000건(5대 사고 유형). 정답 문장은 라벨에서 템플릿으로 생성했습니다.
- 벤치마크는 기본 모델과 LoRA 모델을 비교하며, 학습에 쓴 샘플은 기본으로 평가에서 제외합니다. 결과는 아직 공개하지 않았습니다.
- 라벨을 프롬프트에 넣는 데모 기능은 기본으로 꺼져 있으며, 켠 상태의 결과는 성능 평가에 쓰면 안 됩니다.
