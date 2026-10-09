# Cosmos Guardian 0.2

기존 현장 사진을 **평가서 작성 → 관리감독자 확인 → 개선조치·재확인**으로 연결하는 산업안전 업무지원 제품의 개발 버전입니다. GPU 없이 직접 작성할 수 있으며, NVIDIA Cosmos Reason 2의 실험용 안전모 판별은 선택 기능입니다.

**현장 실증 전입니다.** 현재 제품의 AI 범위는 사진 단위 안전모 미착용 의심 분류입니다. 사람별 검출 박스, 전체 현장 안전 판정, 사고 예측, 실시간 영상 관제 성능은 검증하지 않았습니다. 위험도와 최종 조치는 사람이 결정합니다.

[제품 요구사항과 개발 순서](PRD.md)는 1인 개발 기준으로 정리했습니다. 다음 목표는 사진에서 관찰 가능한 사실·공간 관계·판단 유보의 평가, Cosmos와 다른 시각언어 모델의 동일 과제 비교, 실패 분석에 따른 개선, 전체 업무 시연입니다. 상황 해석은 아직 개발 과제이며 안전모 객체 검출기 비교는 보조 실험입니다. 실제 담당자 참여는 현재 개발의 선행 조건이 아니며, 현장 업무시간 절약은 별도 검증 과제로 남깁니다.

## 제품 방향과 구현 범위

제조사가 이미 가진 CCTV·점검 사진·업무 절차를 활용하고, 검토와 기록에 드는 시간을 줄이는 방향입니다. 기존 시스템의 대체 비용이나 사고 감소율을 가정하지 않습니다.

[Intenseye의 조치 관리](https://www.intenseye.com/products/software-actions)와 [Protex AI의 EHS 연동](https://www.protex.ai/integrations/ehs-integrations)도 기존 인프라·업무 시스템 연결을 설명합니다. 따라서 이 기능 자체를 독창성으로 주장하지 않습니다. 목표 고객을 소규모 제조 사업장으로 좁히고 설치·운영 비용, 기존 양식 전환 시간, 담당자의 처리 시간을 비교해 구매 이유를 입증해야 합니다. 경쟁사 견적과 고객 업무 기준값은 아직 확보하지 못했습니다.

| 비용·업무 문제 | 현재 구현 | 아직 측정·개발할 부분 |
|---|---|---|
| 불필요한 추론과 재전송 | AI 없이 직접 작성, 요청한 사진만 분석, 동일 요청 키 재전송은 기존 작업 재사용 | 영상 변화 감지와 경량 검출기 선별 후 정밀 추론 |
| 분석 중 화면 종료·서버 중단 | SQLite 작업 대기열, 상태 복원, 취소, 제한된 수동 재시도 | 다중 사업장·다중 GPU 운영 |
| AI 결과의 문서 재입력 | 원본·모델 출력에서 평가서 초안 연결, Excel·인쇄용 HTML | 고객 Excel 양식 매핑, 기존 EHS/MES 연동 |
| 경고 후 조치 누락 | 지정 감독자 확인, 담당자·기한·완료·재확인·종결 | 반복 위험 묶기와 기한 초과 알림 |
| 도입 효과를 입증하기 어려움 | 실제 시도 처리시간과 동일 요청 재전송 건수 표시 | 기존 업무 대비 검토 시간·문서 작성 시간·오탐 처리 비용 비교 |

영상 선별, 반복 위험 묶기, 외부 시스템 연동은 **설계 방향이며 구현 완료 기능이 아닙니다**. 현재 동일 요청 키의 중복 방지는 네트워크 재전송에 한정합니다. 다른 키로 제출한 같은 사진, 연속 프레임의 유사 장면을 자동 병합하지 않습니다. 처리시간은 GPU 사용료나 절감액이 아닙니다.

### 실제 검출 성능

[평가 기록](evaluation/README.md)에 데이터, 실행 환경, 실패한 실험까지 보존합니다. 최초 일반 안전 프롬프트의 100장 평가에서는 안전모 유형 정답이 0/50건이었습니다. 이후 안전모 전용 프롬프트와 호환 환경을 검증했습니다. 개발용 20장의 정밀도·재현율은 80%였지만, 별도 공개 평가 100장은 **정밀도 54%·재현율 54%·오탐률 46%**, 형식 오류 0건입니다. 평균 분석 15.036초(p95 29.541초)이며 동시 이미지 빌드의 자원 영향을 포함합니다. **현재 검출기는 제조사 운영용으로 권할 수준이 아닙니다.** 사용자 제공 스마트 야드 100장에서는 **정밀도 72.58%·재현율 90%·오탐률 34%**입니다. 촬영 그룹 2개·연속 묶음 5개의 제한된 자료이므로 공개 결과와 함께 해석해야 합니다. 기본 제공은 GPU 없는 직접 작성·검토이며 AI는 선택적 실험 기능입니다.

Transformers는 **4.57.6**으로 고정합니다. [NVIDIA 공식 의존성](https://github.com/nvidia-cosmos/cosmos-reason2/blob/main/cosmos_reason2_utils/pyproject.toml)은 `>=4.57.0,<5.0.0`을 지정합니다. 앞선 5.1.0 환경의 결과를 호환 환경의 성능으로 일반화하지 않습니다. 프로젝트 LoRA 가중치는 아직 없습니다.

## 실행 구조

```text
브라우저 ── API·정적 화면 (GPU 불필요)
               │
               ├─ SQLite: 작업·평가서·변경 이력
               ├─ uploads: 원본 사진
               │
               └─ 독립 작업자 ── 추론 자식 프로세스 ── GPU
```

API는 모델을 로드하지 않습니다. 작업자는 모델을 재사용하며, 취소·시간 초과 시 추론 프로세스를 종료합니다. 기본 대기열 상한은 20건, 시도 제한은 3회, 시도 제한시간은 300초입니다. 종료된 작업자의 임대 시간이 만료되면 실패로 남기고 사람이 재시도합니다. 실패·판단 불가를 정상 또는 안전으로 바꾸지 않습니다.

### 로컬 실행

Python 3.10+, Node.js, uv가 필요합니다. 실제 추론에는 NVIDIA GPU가 필요합니다. 아래 `.env`의 예시 토큰은 서버가 거부하므로 사용자별로 서로 다른 무작위 24자 이상 토큰을 넣습니다. 토큰은 저장소에 커밋하지 않습니다.

```bash
cp .env.example .env
python -c 'import secrets; print(secrets.token_urlsafe(32))'
# .env에서 사용자 ID와 각 토큰을 수정
uv sync --frozen
cd dashboard
npm ci
npm run build
cd ..
set -a
source .env
set +a
.venv/bin/uvicorn product_server:app --host 127.0.0.1 --port 8888
```

기본 `.env.example`은 AI 분석을 끈 직접 작성·검토 모드입니다. **실험용 AI를 사용할 때만** `uv sync --frozen --extra inference`로 추론 의존성을 설치하고 `.env`의 `COSMOS_INFERENCE_ENABLED=1`로 변경합니다. API를 재시작하고, 다른 터미널에서 프로젝트 루트의 `.env`를 불러온 뒤 작업자를 실행합니다.

```bash
set -a
source .env
set +a
.venv/bin/python worker.py
```

접속 주소는 `http://127.0.0.1:8888`입니다. `/api/docs`에 API 스키마가 있습니다. 모델이 없으면 승인된 Hugging Face 계정으로 `.venv/bin/hf auth login` 후 다운로드합니다. 현재 측정 모델 revision:

```bash
.venv/bin/hf download nvidia/Cosmos-Reason2-2B \
  --revision 9ce19a195e423419c349abfc86fd07178b230561 --cache-dir data/huggingface
```

GPU 없는 직접 작성·검토 실행은 `uv sync --frozen`, `COSMOS_INFERENCE_ENABLED=0` 설정 후 API만 실행합니다. `uv sync`는 선택하지 않은 추론 의존성을 제거하므로 GPU 환경에서는 `--extra inference`를 유지하세요. 연구용 데이터·학습 도구는 `--extra research`를 추가합니다. 프런트엔드만 개발할 때는 `VITE_API_BASE=http://127.0.0.1:8888/api npm run dev`를 사용합니다.

### 컨테이너 설치

기본 직접 작성·검토는 Docker Compose만으로 실행합니다. `.env`의 계정 토큰을 설정하고 `docker compose up --build -d api`를 실행하세요. GPU 분석을 추가할 때는 GPU 컨테이너 실행 환경과 위 모델 다운로드를 준비하고 `.env`에 `COSMOS_INFERENCE_ENABLED=1`을 설정합니다. 모델 캐시는 읽기 전용으로 연결하며, 컨테이너 작업자는 오프라인 모드로 실행합니다.

```bash
docker compose --profile inference up --build -d
docker compose ps
docker compose logs --tail=50 api worker
```

직접 작성·검토만 사용하려면 `.env`에 `COSMOS_INFERENCE_ENABLED=0`을 설정하고 `docker compose up --build -d api`로 실행합니다. API와 작업자는 비루트 사용자로 실행하며 데이터·사진은 별도 볼륨에 남깁니다. 기본 포트는 호스트의 `127.0.0.1:8888`만 연결합니다. 사내 다른 PC에 제공할 때는 HTTPS와 접근 제어를 구성해야 합니다. 계정 설정의 대체 파일은 `COSMOS_ENV_FILE`로 지정할 수 있습니다.

`/api/health`는 API 응답 여부, `/api/ready`는 계정 설정과 작업자 heartbeat를 확인합니다. 작업자 연결은 모델 로딩이나 검출 성능의 보증이 아닙니다. 실제 추론은 첫 접수 때 로드합니다. 작업자를 실행하지 않아도 제출된 작업은 대기 상태로 보존됩니다.

## 관리감독자 검토와 위험성평가서

```text
직접 작성 / 선택적 AI 분석 → 평가서 초안 → 검토 요청 → 승인 → 개선조치 → 재확인 → 종결
                           └→ 보완 요청 → 수정 → 재검토
```

- 직접 작성은 추론 0회로 초안을 만들고 `analysis_performed: false`를 보존합니다. 위험요인 목록은 사람이 채우며 미검출·안전 판정으로 기록하지 않습니다. AI 대기열이 가득 차거나 작업자가 없어도 직접 작성할 수 있습니다.
- 원본 사진 SHA-256, AI 사용 시 모델 출력·revision, 프롬프트 해시, 분석 시간과 작성자를 저장합니다. 원본 조회 시 해시를 확인합니다. 작업 실패 원본도 재시도·검토를 위해 보존합니다.
- 사람이 사업장·공정·작업·평가일·참여자·평가 기준과 위험별 예상 피해·현재 조치·추가 조치·담당자·기한을 입력합니다. 오탐은 삭제하고 누락 위험은 추가할 수 있으며 원래 모델 출력은 남습니다.
- **프로젝트 기본 5×5 양식**입니다. 가능성·중대성은 사람이 각각 1–5로 입력합니다. 특정 기관의 공식 양식이나 법적 적합성 판정은 아닙니다. 안전모 결과를 임의의 0–100 안전점수로 바꾸지 않습니다.
- 서버에 등록한 사용자별 토큰으로 작성자와 감독자를 구분합니다. 지정 감독자만 승인·보완·재확인·종결할 수 있고, 승인에는 원본·위험요인·개선조치·평가 기준의 네 확인 항목과 의견이 필요합니다.
- 승인 전 출력은 차단합니다. 승인 후 본문을 잠그고 조치 이력을 별도로 추가합니다. 증빙은 현재 문서 위치·텍스트이며 증빙 파일 업로드와 승인본 개정은 아직 없습니다.
- Excel에는 개요·위험요인·조치·확인 이력이 포함됩니다. PDF는 내려받은 인쇄용 HTML을 브라우저의 ‘인쇄 / PDF로 저장’으로 생성합니다.
- 변경마다 사용자·역할·시각·버전·스냅샷을 해시 연결 이력으로 남깁니다. 오래된 버전의 수정·승인은 409로 거부합니다. 외부에 보관한 해시가 없으면 DB 전체 이력 재작성이나 꼬리 삭제를 탐지할 수 없습니다. 인증서 기반 전자서명이 아닙니다.

현재 **단일 사업장** 모델입니다. 등록 사용자는 같은 현장의 평가서를 열람·작성하며, 작업 목록은 작성자 본인 또는 감독자가 조회합니다. 조직별 분리, SSO, 비밀번호 재설정·토큰 발급 관리 화면은 없습니다. 개인 토큰을 공유하지 마세요. 대시보드 토큰은 브라우저 메모리에만 유지합니다.

### 주요 API (접두사 `/api`)

`health`, `ready` 외에는 `Authorization: Bearer <token>`이 필요합니다.

| API | 용도 |
|---|---|
| `GET /me`, `GET /supervisors` | 본인 역할·감독자 목록 |
| `POST /jobs` | 이미지 접수, `mode=manual` 또는 `helmet`, `Idempotency-Key` 필수, 202 반환 |
| `GET /jobs`, `GET /jobs/{id}` | 작업 상태·시도·평가서 연결 |
| `POST /jobs/{id}/{cancel,retry}` | 작성자 취소·수동 재시도 |
| `GET /assessments`, `GET /assessments/{id}` | 평가서·모델 원본·이력 |
| `PUT /assessments/{id}` | 초안 수정 (`version`, `form`) |
| `POST /assessments/{id}/submit` | 검토 요청 |
| `POST /assessments/{id}/decisions/{approve,request_changes,close}` | 감독자 결정 |
| `POST /assessments/{id}/actions/{row}/{complete,verify,reopen}` | 조치 완료·재확인·재조치 |
| `GET /assessments/{id}/{media,integrity}` | 원본·이력 검증 |
| `GET /assessments/{id}/export/{xlsx,html}` | 승인본 출력 |

기존 `POST /analyze` SSE는 실험 기능으로 기본 차단됩니다. 제품 대기열과 함께 활성화하면 독립 추론 경로가 GPU를 공유하므로 제품 운영에는 사용하지 않습니다.

## 백업·복원

SQLite의 일관된 스냅샷과 작업·평가서가 참조하는 사진을 함께 백업하고 파일별 SHA-256을 기록합니다. 복원은 존재하지 않는 새 디렉터리에만 가능하며 모든 해시를 먼저 확인합니다.

```bash
.venv/bin/python scripts/product_backup.py backup \
  --database data/workflow.sqlite3 --uploads uploads --destination outputs/backup-001
.venv/bin/python scripts/product_backup.py restore \
  --archive outputs/backup-001 --destination outputs/restored-001
```

복원본으로 전환할 때 API·작업자를 중지하고 DB·업로드 경로를 함께 변경합니다. 계정 비밀값·모델 가중치·선택적 기존 JSONL 분석 로그는 이 백업에서 제외되므로 별도로 보관합니다. 컨테이너에는 같은 도구가 포함됩니다. 볼륨 내 백업만으로는 호스트 장애에 대비할 수 없으므로 백업 폴더를 별도 저장소로 복사해야 합니다.

## 검증

2026-10-09 기준 Python 테스트 99개와 브라우저 테스트 4개가 통과했습니다. API 이미지 빌드는 성공했지만, 저장 공간 문제로 최종 GPU 작업자 이미지 빌드를 중단했습니다. 아래의 실제 컨테이너 통합 확인(`scripts/smoke_product.py`)은 아직 실행하지 않았습니다. 따라서 이 버전은 배포 검증을 마친 정식 제품이 아닌 개발 중인 MVP입니다.

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check core tests server.py product_server.py worker.py \
  scripts/benchmark.py scripts/prepare_public_ppe_eval.py scripts/prepare_aihub_helmet_eval.py scripts/product_backup.py scripts/smoke_product.py
cd dashboard
npm run lint
npm run build
npm run test:e2e
```

업무·브라우저 테스트는 분리된 DB와 명시적인 모의 추론을 사용하며 검출 성능 증거가 아닙니다. 실제 GPU 지표는 `scripts/benchmark.py --profile helmet`로 측정합니다. API와 작업자 이미지를 각각 `cosmos-guardian-api:dev`, `cosmos-guardian-worker:dev`로 빌드한 뒤 실제 연결을 확인하는 명령:

```bash
.venv/bin/python scripts/smoke_product.py --image <실제-평가-이미지.png>
```

이 확인은 임시 계정·볼륨에서 GPU 없는 직접 작성과 실제 한 장 추론, 중복 요청, API 재시작 후 작업 보존, 원본 조회, 백업·복원을 확인하고 임시 컨테이너·볼륨을 정리합니다. 감독자 승인이나 현장 실증 결과를 만들지 않습니다.

## 연구용 구성과 사용 조건

`app.py`, `streamlit_app.py`, `utils/live_feed.py`는 기존 일반 안전 추론 실험용입니다. `prompts/safety_inspector.yaml`과 느슨한 JSON 복구는 이 경로에만 적용됩니다. 제품 안전모 경로는 정확한 YES/NO/UNKNOWN 응답만 허용하며 형식 오류를 임의 복구하지 않습니다. 라벨 주입은 제품·벤치마크에서 금지됩니다.

AI Hub 변환 학습 JSON은 스마트팩토리 끼임 105건, 건설 1,000건입니다. 원본 미디어와 학습된 LoRA는 저장소에 포함하지 않습니다. 사용자 제공 스마트 야드 ZIP에서 원본·라벨 100쌍을 별도 평가용으로 추출했습니다. 템플릿 정답은 전문가 추론의 검증 자료가 아닙니다. 추가 공개 안전모 데이터의 출처·다운로드·분리 기준은 [평가 문서](evaluation/README.md)에 있습니다.

코드는 Apache-2.0, 모델은 NVIDIA의 모델 사용 조건, 데이터는 각각의 출처 사용 조건을 따릅니다. 상용 도입 전 남은 작업은 고객 현장 검증, 오탐·누락 기준 합의, 접근 관리, 보존 정책, 실제 양식·시스템 연동, 운영 지원 체계와 비용 효과 측정입니다.
