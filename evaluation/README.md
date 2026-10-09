# 실제 벤치마크 실행

이 폴더에는 평가 입력 형식, 공개 안전모 데이터 총 220장의 목록, 실제 모델 실험 기록이 있습니다. `manifest.example.json`은 형식 예시이며, 원본 이미지는 Git에서 제외한 `data/public_ppe/images/`와 `data/public_ppe_v1/images/`에 저장됩니다.

## 안전모 제품 프로필 개발·별도 평가 (2026-10-09)

최초 평가 이후 **120장**을 추가 다운로드했습니다. 개발용 `public_ppe_dev_v1.json` 20장(10/10, seed 1042), 별도 평가용 `public_ppe_test_v1.json` 100장(50/50, seed 2042)입니다. 기존 100장·개발·평가 사이 ID와 SHA-256 중복을 제거했습니다. 동일 공개 원천의 데이터이므로 촬영 현장·세션의 독립성이 입증된 외부 검증이라고 부르지 않습니다.

제품 프로필은 사진에서 미착용자가 있는지 YES/NO/UNKNOWN으로 답하게 합니다. 정확한 분류 응답만 받아 `ppe_violation`으로 변환하며 전체 현장 안전점수나 사람별 박스는 만들지 않습니다. UNKNOWN은 판단 유보로 따로 집계하고 착용 예측으로 계산하지 않습니다. 모델에 정답 라벨을 전달하지 않습니다.

개발 실험의 순서는 다음과 같습니다. 앞선 실패 결과도 보존합니다.

| 개발 실험 | 20장 결과 | 의미 |
|---|---|---|
| JSON 응답, Transformers 5.1.0 | 형식 오류 20/20 | 해당 출력 계약 실패 |
| 짧은 분류 응답, Transformers 5.1.0 | TP 0, FP 0, TN 10, FN 10 | 형식 성공만으로 검출 성능이 개선되지 않음 |
| 동일 분류 프롬프트, Transformers 4.57.6 | TP 8, FP 2, TN 8, FN 2 | 개발용 정밀도·재현율 각각 80%, 평균 11.766초 |

개발 결과 원본: [JSON 응답 실패](results/20261009T041054Z-8a753933/benchmark_results.json), [5.1.0 분류 응답](results/20261009T041919Z-1b6b0506/benchmark_results.json), [4.57.6 분류 응답](results/20261009T065145Z-d113af66/benchmark_results.json). 각 폴더에 실행 시점 소스와 manifest를 보존합니다. 이 20장 결과는 별도 평가 점수가 아닙니다.

[NVIDIA 공식 의존성](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/cosmos_reason2_utils/pyproject.toml)은 Transformers `>=4.57.0,<5.0.0`을 지정합니다. 같은 개발 목록·프롬프트의 재실행에서 버전 변경 후 판별 결과가 달라졌습니다. 의존성 변경은 huggingface-hub 등도 함께 바꾸므로 Transformers 내부의 특정 버그까지 원인으로 확정하지 않습니다. 별도 평가 100장은 개발 실험이 끝난 뒤 고정된 분류 프롬프트와 4.57.6으로 실행합니다.

재현 명령:

```bash
.venv/bin/python scripts/prepare_public_ppe_eval.py \
  --output evaluation/public_ppe_dev_v1.json --media-dir data/public_ppe_v1/images \
  --per-class 10 --seed 1042 --exclude-manifest evaluation/public_ppe_manifest.json
.venv/bin/python scripts/prepare_public_ppe_eval.py \
  --output evaluation/public_ppe_test_v1.json --media-dir data/public_ppe_v1/images \
  --per-class 50 --seed 2042 --exclude-manifest evaluation/public_ppe_manifest.json \
  --exclude-manifest evaluation/public_ppe_dev_v1.json
.venv/bin/python scripts/benchmark.py --profile helmet --base-only \
  --manifest evaluation/public_ppe_test_v1.json \
  --development-manifest evaluation/public_ppe_dev_v1.json \
  --output-dir outputs/helmet_product_benchmark
```

`--development-manifest`는 개발·평가 ID와 파일 해시 중복을 검사합니다. 원천에 촬영 세션 정보가 없으므로 그룹 독립성이나 모델 사전학습 데이터 중복 없음까지 보증하지 않습니다. 제품 작업자는 평가에서 사용한 모델 revision으로 고정합니다. `--profile legacy`는 기존 일반 안전 추론 경로입니다.

## 별도 공개 평가 100장 완료

실행 `20261009T074129Z-27083abe`, 안전모 프로필, Transformers 4.57.6, RTX 3090, 고정 모델 revision `9ce19a195e423419c349abfc86fd07178b230561`입니다. 개발 자료와 ID·파일 내용이 겹치지 않는 100장을 모두 추론했습니다.

| 항목 | 실측 |
|---|---:|
| 미착용 검출 TP / 누락 FN | 27 / 23 |
| 착용 오탐 FP / 올바른 착용 TN | 23 / 27 |
| 정밀도 / 재현율 | **54% / 54%** |
| 착용 장면 오탐률 | **46% (23/50)** |
| 출력 형식 오류 / 판단 유보 | 0 / 0 |
| 전체 이미지 정답률 | 54/100 |
| 분석 평균 / p95 | 15.036 / 29.541초 |

개발 20장의 80%를 최종 성능으로 제시하지 않습니다. **현재 프로필을 제조사의 운영용 검출기로 권할 수 없습니다.** 업무 흐름·배포 테스트 통과와 검출 성능은 별개입니다. 원본 파일·라벨·순서, 소스 스냅샷과 요약 지표의 재계산을 검증했습니다.

[결과 원본](results/20261009T074129Z-27083abe/benchmark_results.json) · [검증 기록](results/20261009T074129Z-27083abe/verification.json) · [실행 소스](results/20261009T074129Z-27083abe/source/)

같은 호스트에서 Docker 이미지 빌드가 진행됐으므로 지연시간은 이 실행에서 관측한 값이며 격리된 최대 처리량 측정은 아닙니다. 모델 로딩·업로드·대기열 시간은 제외합니다. 앞선 `20261009T072627Z-19ece6ba` 실행은 23장 후 종료 신호(exit 143)로 중단됐으며, [중간 기록](results/20261009T072627Z-19ece6ba/benchmark_results.json)의 `running`은 마지막 체크포인트 상태입니다. 살아 있는 프로세스나 완료 결과가 아닙니다. 중단 기록을 보존하고 결과에 맞춰 프롬프트를 조정하지 않은 채 전체 100장을 다시 실행했습니다.

## 2026-10-09 실측 결과 (최초 일반 안전 프로필)

실행 ID `20261009T030509Z-efdf529e`. NVIDIA Cosmos-Reason2-2B 기본 모델, RTX 3090, BF16, Python 3.10.12, torch 2.9.0+cu128, transformers 5.1.0으로 실제 이미지 100장을 모두 추론했습니다. 모델 revision은 `9ce19a195e423419c349abfc86fd07178b230561`입니다. LoRA와 정답 라벨 주입은 사용하지 않았습니다. 서비스 에이전트의 greedy 생성 설정(`max_new_tokens=1536`, `repetition_penalty=1.1`)과 기존 물리 설명 중심 프롬프트를 평가 중 변경하지 않았습니다.

| 항목 | 실측 |
|---|---:|
| 안전모 미착용 / 착용 표본 | 50 / 50장 |
| 평가 가능한 출력 / 출력 형식 오류 | 86 / 14건 |
| 미착용을 `ppe_violation`으로 검출 | **0/50건 (0%)** |
| 위험유형 집합 완전 일치 (오류 포함 전체 분모) | **18/100건 (18%)** |
| 착용 장면의 범위 밖 위험 경고 (유효 출력 기준) | 26/44건 (59.09%) |
| 성공한 분석 평균 / p95 지연시간 | 22.486 / 52.034초 |
| 실패 포함 전체 시도 평균 / p95 지연시간 | 27.001 / 69.845초 |

일반적인 경고 유무와 안전모 유형 정답은 다릅니다. 비어 있지 않은 위험 목록 40건은 **모두 `struck_by`로 분류**됐고, `ppe_violation` 출력은 없었습니다. 따라서 다음 표의 미착용 장면 경고 14건을 안전모 검출 정답으로 해석하면 안 됩니다.

| 출처 라벨 | 임의의 위험 경고 있음 | 경고 없음 | 출력 형식 오류 |
|---|---:|---:|---:|
| 안전모 미착용 50장 | 14 | 28 | 8 |
| 안전모 착용 50장 | 26 | 18 | 6 |

원시 요약의 binary precision 35%, 유효 출력 recall 33.33%, 실패 포함 effective recall 28%는 **위험 종류와 무관한 경고 유무** 지표입니다. 안전모 유형 검출률은 0%입니다. 현재 설정의 기본 모델을 안전모 착용 판단에 사용할 성능 근거가 없습니다. 작업 절차·양식 연결의 테스트 통과는 모델의 안전 성능을 보증하지 않습니다.

결과 상태는 `completed_with_errors`이며 종료 코드는 1입니다. 이는 100장 중 14장의 출력 오류를 알리며, 실행 중단이나 샘플 누락을 뜻하지 않습니다. 성공한 분석의 지연시간에는 전처리·생성·파싱이 포함됩니다. 실패 포함 시도 시간 합계는 2,700.094초이며, 가장 오래 걸린 실패는 105.987초입니다. 모델 다운로드·로딩, 시도 사이 저장, 화면 표시·업로드는 이 시간에서 제외합니다.

- [측정 결과 원본 JSON](results/20261009T030509Z-efdf529e/benchmark_results.json): 샘플별 정답·파싱된 출력·오류·시간과 실행 환경. 생성 원문 전체는 보존하지 않았습니다.
- [검증 및 파생 지표](results/20261009T030509Z-efdf529e/verification.json): 요약 재계산 일치, 100장 순서·라벨·파일 해시 일치, 실행 코드·프롬프트·manifest 스냅샷 해시, 실패 포함 지연시간.
- [실행 시점 소스 사본](results/20261009T030509Z-efdf529e/source/): 수정된 작업 트리에서 실행했으므로 해당 파일 바이트를 함께 보존했습니다. 평면 파일 사본은 출처 확인용이며 독립 실행 가능한 체크아웃은 아닙니다.

이 최초 실행의 범위는 아래의 공개 안전모 데이터에 한정됩니다. 전문가 독립 재검수, 원본 촬영 세션 분리, 사전학습 중복 검증은 완료되지 않았습니다. 당시에는 AI Hub 현장 데이터 평가가 없었으며, 이후 수행한 국내 스마트 야드 평가는 다음 절에 별도로 기록했습니다. LoRA 비교와 물리 설명의 타당성 검증은 아직 없습니다. 모델 개선 후 성능을 주장하려면 이번 결과를 보존하고 별도 개발 데이터와 새 평가 데이터로 검증해야 합니다.

## 국내 현장 데이터: 사용자 제공 자료 연결 완료

사용자가 제공한 `D:\081.선박·해양플랜트 스마트 야드 안전 데이터`에서 ZIP 77개(약 22.69 GiB)를 확인했습니다. 출처는 [AI Hub 스마트 야드 안전 데이터 (71770)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71770)입니다. 안전장비 `S63_DATA3`의 안전모 미착용 `H0`·착용 `H1` 원천 JPG와 대응 JSON이 있습니다. 원래 D 드라이브 파일은 수정하지 않습니다.

- H0: Training 13,502쌍, Validation 1,688쌍.
- H1: Training 13,559쌍, Validation 1,695쌍.
- Validation에서 각 50장, 총 100장을 `aihub_helmet_test_v1.json`으로 준비했습니다. 폴더명만 믿지 않고 객체 클래스·착용 카테고리·이미지 파일명을 대조합니다. 유효한 머리 객체 라벨이 없는 자료 등 제외 사례를 manifest에 남깁니다.
- 모든 ZIP을 풀지 않고 선택한 파일만 CRC를 검증해 `data/aihub_helmet_v1/images/`와 `labels/`에 읽어 둡니다. 이미지·라벨·원본 ZIP·준비 코드 해시를 기록합니다.
- 선택 자료는 **촬영일·위치 코드 기준 2개 그룹, 연속 촬영 5개 묶음**입니다. `L1:2023-10-25` 그룹은 원래 Training에도 있어 공식 분할명만으로 세션 독립성을 인정할 수 없습니다. 이 프로젝트는 해당 자료로 학습하지 않았습니다. 앞으로 이 자료를 학습에 사용하려면 전체 학습 목록을 제공하고 촬영 그룹이 겹치는 평가 자료를 교체해야 합니다.
- 이 평가는 국내 조선 현장 이미지의 범위 전이 확인입니다. 서로 독립인 제조 현장 100곳의 검증, 객체 위치 mAP, 모든 제조 업종의 실증으로 해석하지 않습니다.

```bash
.venv/bin/python scripts/prepare_aihub_helmet_eval.py \
  --source-root '/mnt/d/081.선박·해양플랜트 스마트 야드 안전 데이터'
.venv/bin/python scripts/benchmark.py --profile helmet --base-only \
  --manifest evaluation/aihub_helmet_test_v1.json \
  --development-manifest evaluation/public_ppe_dev_v1.json \
  --output-dir outputs/aihub_helmet_benchmark
```

다른 환경에서 원본이 필요하면 공식 페이지에 본인 계정으로 로그인해 ‘다운로드’를 신청하고, 승인 후 ‘파일 목록 (API 다운로드)’에서 H0·H1 원천·라벨을 함께 받습니다. 키를 코드나 채팅에 기록하지 않습니다. 기존 `download_aihub_real.py`의 추정 API 경로는 이번 수령에 사용하지 않았습니다.

## 국내 스마트 야드 100장 실측 완료

실행 `20261009T081225Z-0d1392c5`. 공개 평가 이후 모델·프롬프트·생성 설정을 변경하지 않고 사용자 제공 자료를 추론했습니다.

| 항목 | 실측 |
|---|---:|
| 미착용 검출 TP / 누락 FN | 45 / 5 |
| 착용 오탐 FP / 올바른 착용 TN | 17 / 33 |
| 정밀도 / 재현율 | **72.58% / 90%** |
| 착용 장면 오탐률 | **34% (17/50)** |
| 출력 형식 오류 / 판단 유보 | 0 / 0 |
| 전체 이미지 정답률 | 78/100 |
| 분석 평균 / p95 | 11.188 / 15.442초 |

[결과 원본](results/20261009T081225Z-0d1392c5/benchmark_results.json) · [검증 및 그룹별 결과](results/20261009T081225Z-0d1392c5/verification.json) · [자료 준비 코드](results/20261009T081225Z-0d1392c5/data-preparation/prepare_aihub_helmet_eval.py)

100쌍의 원본·라벨 해시와 정답·순서, 실행 소스, 요약 재계산을 확인했습니다. 공개 데이터 220장과 정확한 파일 중복도 없습니다. 다만 **촬영 그룹 2개·연속 묶음 5개**의 상관된 프레임이며 앞서 명시한 공식 분할 중복이 있습니다. 높은 재현율만 강조하거나 공개 평가의 54% 결과를 생략하지 않습니다. 오탐 17건과 누락 5건이 있어 제조사의 운영용 검출기로 권할 수준은 아닙니다. 같은 호스트의 컨테이너 작업 영향을 포함한 관측 지연시간이며 모델 로딩·업로드·대기시간은 제외합니다.

## 데이터 준비

1. 학습에 사용하지 않은 실제 이미지·영상에 독립적으로 정답을 부여합니다. 위험 장면과 정상 장면을 모두 포함합니다.
2. 예시를 `evaluation/manifest.json`으로 복사하고 실제 경로·정답·검토자 정보를 입력합니다. 미디어 경로는 manifest 파일 기준입니다.
3. LoRA 비교 시 해당 어댑터에 사용한 **전체 학습 데이터**를 같은 형식의 `evaluation/training.json`으로 정리합니다. 정상 장면은 `is_hazardous: false`, `events: []`로 표시합니다.
4. `group_id`는 원본 촬영 세션·사건 단위로 지정합니다. 동일 사건의 다른 카메라, 연속 프레임, 잘라낸 영상은 같은 그룹이어야 합니다. 임의로 다른 ID를 부여하면 그룹 누출 검증이 무력화됩니다.

필수 항목: `id`, `group_id`, `media_path`, `is_hazardous`, `events`, `annotation_source`.
사고 유형은 `entrapment`, `fall`, `struck_by`, `ppe_violation`, `other`입니다. AI의 자유 서술에 단어가 등장하는 것만으로 정답 처리하지 않고 `hazards_detected[].event_type`과 정확히 비교합니다. 여러 위험은 여러 라벨로 표시합니다.

## 사전 확인과 실행

프로젝트 루트에서 실행합니다. 실측에는 추론 의존성과 NVIDIA GPU가 필요합니다. GPU 환경에서는 선택적 의존성을 유지합니다.

```bash
uv sync --frozen --extra inference
.venv/bin/python scripts/benchmark.py \
  --manifest evaluation/manifest.json \
  --training-manifest evaluation/training.json \
  --adapter outputs/cosmos-reason2-2b-safety-lora \
  --preflight

.venv/bin/python scripts/benchmark.py \
  --manifest evaluation/manifest.json \
  --training-manifest evaluation/training.json \
  --adapter outputs/cosmos-reason2-2b-safety-lora
```

`--preflight`는 모델을 로드하지 않으며 `outputs/benchmark_results/preflight.json`에 준비 상태를 저장합니다. 데이터·의존성·어댑터가 빠지면 종료 코드 2를 반환합니다. GPU 사용 가능 여부와 모델 다운로드·로딩 성공은 실제 실행에서 확인합니다.

기본 모델만 평가하려면 명시적으로 `--base-only`를 사용합니다. 이 모드에는 LoRA 비교 결과가 없습니다. 학습 manifest를 생략하면 그룹·내용 해시 기준 학습 데이터 중복 검증이 수행되지 않았다고 결과에 표시합니다. 저장소 학습 JSON의 샘플 ID는 항상 중복 검사에 사용합니다.

LoRA 비교는 어댑터 설정과 가중치 파일이 모두 필요하며 로드 실패 시 중단합니다. 학습·평가 간 ID, 그룹, 미디어 SHA-256 중복을 거부합니다. 검사 범위는 제공된 학습 manifest에 한정되므로 누락 없는 학습 목록과 독립 라벨링이 필요합니다. 모델 사전학습 데이터와의 중복까지 검증하는 기능은 아닙니다.

## 결과 해석

실측 결과는 실행별 `outputs/benchmark_results/<UTC 시각>-<실행 ID>/benchmark_results.json`에 저장됩니다. 같은 서비스 에이전트와 프롬프트를 사용하고 라벨 주입은 끕니다. 정답 라벨은 모델에 전달하지 않습니다.

- TP/FP/TN/FN, 위험·정상 장면별 출력 실패 수
- 유효 출력의 precision, recall, 정상 장면 오탐률
- 실패한 위험 장면까지 분모에 포함한 effective recall
- 사건 라벨 집합의 완전 일치율, 유형별 TP/FP/FN
- 유효 출력 비율, 성공한 분석의 평균 및 p95 지연시간
- 샘플별 정답·출력·오류와 입력 해시
- Git 커밋/수정 상태, 모델 revision, 어댑터·프롬프트·코드·manifest 해시, 라이브러리 및 GPU 정보

출력 실패를 정상 예측이나 0초 추론으로 계산하지 않습니다. 분모가 없으면 지표는 `null`입니다. 지연시간은 모델 로딩을 제외한 전체 에이전트 분석 시간이며, 화면 표시·업로드 시간은 제외합니다. 모델 JSON이 복구된 경우도 스키마를 만족하면 유효 출력으로 계산합니다. 이 평가는 위험 검출과 사건 분류를 측정하며 물리 설명의 타당성이나 현장 위험도 점수의 적정성을 검증하지 않습니다.

`completed_with_errors` 또는 `failed` 결과는 오류를 포함합니다. 실험 결과를 보고할 때 표본 수·사고 유형·정상/위험 비율과 오류 건수를 함께 제시해야 합니다. 테스트용 모의 추론 결과는 이 폴더에 벤치마크로 저장하지 않습니다.

## 공개 안전모 평가 데이터 (100장)

출처는 [Voxel51/hard-hat-detection](https://huggingface.co/datasets/Voxel51/hard-hat-detection)이며 revision `d11e7d6b69ba1a22422db3230c0415d39a699d7c`를 고정했습니다. 출처는 helmet/head/person 박스 라벨과 CC0-1.0을 명시합니다. 생성 스크립트는 원본 `head`가 있으면 미착용, `helmet`만 있으면 착용으로 집계하고 `person` 등 다른 라벨이 있거나 어려운 객체로 표시된 이미지는 제외합니다. seed 42로 각 50장을 선택하고 내용이 완전히 같은 파일은 중복 제거합니다.

이는 **안전모 착용 여부 평가**입니다. 착용 클래스는 현장 전체가 안전하다는 뜻이 아닙니다. 기존 라벨의 독립 전문가 재검수나 원본 촬영 세션 정보는 제공되지 않았으므로 그 한계를 manifest에 기록했습니다. 모든 이미지에는 동일한 보수적 `source-session-unknown` 그룹을 부여했습니다. 이 데이터를 학습용에도 사용하면 그룹 중복 검사에서 거부됩니다. 원본 데이터셋의 단일 split 이름은 `train`이지만, 현재 프로젝트는 선택한 100장을 학습에 사용하지 않습니다. 모델 사전학습 포함 여부와 유사 이미지는 검증되지 않았습니다.

재생성 및 준비 확인:

```bash
.venv/bin/python scripts/prepare_public_ppe_eval.py
.venv/bin/python scripts/check_runtime.py
.venv/bin/python scripts/benchmark.py \
  --manifest evaluation/public_ppe_manifest.json --base-only --preflight \
  --output-dir outputs/public_ppe_benchmark
```

`check_runtime.py`는 실제 GPU BF16 행렬 연산, 추론 모듈 import, 공식 모델 설정 다운로드 권한을 확인하고 `outputs/runtime-readiness.json`에 저장합니다. 이 GPU 확인은 모델 추론이나 안전 성능 벤치마크가 아닙니다.

2026-10-09 확인: 추론 의존성 설치와 RTX 3090/CUDA 12.8의 BF16 연산, 승인된 Hugging Face 계정을 통한 모델 다운로드와 실제 추론이 모두 성공했습니다. 다른 환경에서 재현할 때는 [공식 모델 페이지](https://huggingface.co/nvidia/Cosmos-Reason2-2B)에서 사용자 본인이 사용 승인을 받고, 프로젝트 루트의 터미널에서 다음 명령으로 로그인합니다. 토큰을 코드나 채팅에 기록할 필요는 없습니다.

```bash
.venv/bin/hf auth login
```

승인된 계정으로 로그인한 뒤 다음 명령이 실제 모델을 다운로드하고 100장 추론을 실행합니다. 이 명령은 GPU 접근이 가능한 환경에서 실행해야 합니다.

```bash
HF_HUB_CACHE="$PWD/data/huggingface" .venv/bin/python scripts/benchmark.py \
  --manifest evaluation/public_ppe_manifest.json --base-only \
  --output-dir outputs/public_ppe_benchmark
```

공통 평가 맥락은 manifest의 `safety_context`에서 읽습니다. 샘플마다 다른 정답을 프롬프트에 넣지 않습니다. 원본 파일 내용이 manifest의 `media_sha256`과 다르면 실행을 거부합니다. 기본 모델 평가만 가능하며 LoRA 성능 개선을 주장하지 않습니다.

벤치마크는 `HF_HUB_CACHE`가 없으면 프로젝트의 `data/huggingface`를 사용합니다. 서버에서도 같은 모델 파일을 사용하려면 `.env.example`의 캐시 설정을 불러옵니다. 결과는 샘플마다 원자적으로 저장되며, 실행 중인 파일은 `status: running`과 `progress`로 구분합니다. 이번 완료 결과는 무시되는 `outputs/` 경로 외에 위 `evaluation/results/`에도 원본 그대로 보존했습니다.
