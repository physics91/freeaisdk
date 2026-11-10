<!-- regular-plan-v1 -->
<!-- archived: P0–P2 complete; kept for history only -->
# freeai 로드맵 구현 (P0 → P1 → P2)

## Goal

`docs/research/2026-11-06-free-llm-sdk-improvements/report.md`의 로드맵 P0 9건, P1 8건, P2 4건을 순서대로 구현한다.
각 단계가 끝날 때마다 테스트, ruff, mypy strict, 실제 네트워크 검사(키 없이 가능한 범위)를 통과시킨다.

## Scope

- 포함: `src/freeai/` 전체, `tests/`, `README.md`, `pyproject.toml`, 새 모듈(`state.py`, `_core.py`, `aio.py`,
  `compat.py`, `server.py`, `otel.py`, `__main__.py`), `.github/workflows/` 파일 작성(로컬 파일만).
- 제외: PyPI 배포, GitHub 원격 생성, 실제 API 키가 필요한 실측(키가 없음), Redis 백엔드(인터페이스만).

## Approach (결정 사항)

- 실패 정보에 `scope`(model, provider, request)를 명시하고, `QUOTA_EXHAUSTED`, `NOT_ENTITLED`,
  `CONTEXT_EXCEEDED`, `OUTPUT_LIMIT` 종류를 추가한다. 일일 한도는 리셋 시각까지 정지한다.
- 공급자별 차이는 `Provider`의 훅으로 둔다: 오류 해석(`_classify_details`), 요청 정리(`prepare_payload`),
  리셋 경계(`reset_tz`), 힌트로 쓸 리셋 헤더(`hint_headers`), 무료 등급(`tier`), 학습 여부(`trains_on_prompts`).
- `from_env()`는 기본으로 `tier="free"` 공급자만 켠다. Cerebras는 `trial`, Mistral은 `paid`.
- 제한기는 모델별, 공급자별 다중 창(GCRA 방식 RPM/TPM, 경계 고정 창 RPD/월)으로 바꾸고 응답 헤더로 보정한다.
- 상태는 `StateStore` 프로토콜(memory, SQLite WAL)에 epoch 시계로 저장한다.
- async는 I/O 없는 계획/반영 로직(`_core.py`)을 공유하고 `AsyncFreeAI`가 `httpx.AsyncClient`를 쓴다.
- HTTP 의존은 `_http.py` 경계 뒤로 모으고, P2-4에서 httpx2 전환 여부를 검증 후 결정한다.
- 라이선스는 사용자가 정하지 않았으므로 MIT를 기본으로 두고 보고서에 확인 필요로 남긴다.
- 백업: 단계마다 scratchpad `snapshots/`에 사본을 남긴다(Git 저장소가 아님).

## Steps

1. P0-1 ~ P0-9: 분류 순서, 본문 신호, 힌트 범위, 오류 본문 정규화, 403/404 세분화, 공급자 단위 400,
   추론 예산 소진, 무료 보장 기본값, 프리셋 데이터와 `NullHandler`, `sync_quota` 자동 호출.
2. P1-1 ~ P1-8: 다중 창 제한기, 신규 프리셋(Cloudflare, Z.ai, Kilo, NVIDIA, Requesty, 키 없는 공급자),
   `StateStore`, `AsyncFreeAI`, `strategy="health"`와 반개방 확인과 재시도 예산, 스트림 강화(SSE 명세 파서,
   타임아웃, deadline), 개인정보 필터, `reject_response` 훅.
3. P2-1 ~ P2-4: OpenAI 형태 facade, `freeai serve` 로컬 프록시, `freeai[otel]`와 별칭, 패키징(httpx2 결정,
   uv_build 상한, PEP 639, Trusted Publishing 워크플로).

## Verification

- `uv run pytest`(모의 HTTP), `FREEAI_LIVE=1 uv run pytest tests/test_live.py`(실제 OpenRouter, 키 없이 가능한 범위)
- `uvx ruff check src tests`, `uvx ruff format --check src tests`, `uv run --with mypy mypy`
- 프록시는 로컬에서 실제 소켓으로 띄워 OpenAI 형식 요청을 보내 확인한다.
- 실제 채팅 경로는 키 없는 OVHcloud 익명 등급으로 검증한다(동기 chat, async stream 각 1회).
  키가 필요한 공급자의 채팅과 플랫폼별 429 실측은 API 키가 없어 미검증으로 보고한다.

## Progress

- P0 완료: 70 passed, ruff, mypy strict 통과. 스냅샷 `snapshots/01-p0`.
- P1 완료: 103 passed, ruff, mypy strict 통과. 실제 경로: OpenRouter 카탈로그와 401, Kilo와 Requesty
  익명 카탈로그 파싱, OVH 익명 chat(`gpt-oss-120b` → `pong`)과 async stream 성공. 스냅샷 `snapshots/02-p1`.
  - 구조: `_core.py`(Router, Plan, I/O 없음), `client.py`(FreeAI), `aio.py`(AsyncFreeAI), `state.py`.
  - 한도: `Limit`(분/시간 GCRA, 일/월 고정 창), 응답 헤더 `RateSnapshot`이 카운터를 덮어씀. 고정 창은
    서버가 준 `reset_in`이 있으면 달력 경계 대신 그 시각을 씀(Groq 일일 창은 달력 기준이 아님).
  - 상태 키는 `state_id`(공급자 이름 + API 키 SHA-256 앞 12자)로 계정별 분리.
  - 동작 변경: `clock` 기본값이 `time.time`(epoch). 단조 시계를 넘기면 안 됨.
  - 프로세스 로컬: health 통계, 재시도 예산, half-open 탐침, `max_concurrency` 슬롯.
- P2 완료: OpenAI 형태 facade(`compat.py`), 로컬 프록시(`server.py`, `freeai serve`), CLI(`__main__.py`),
  별칭(`fast`, `smart`, `coding`, `auto`), OTel 매핑(`otel.py`, `freeai[otel]`), PEP 639 MIT와 `LICENSE`,
  `.github/workflows/ci.yml`과 `publish.yml`(Trusted Publishing, 로컬 파일만). 실제 경로: 빌드한 wheel을 격리 설치해
  CLI로 OVH 채팅, 실제 소켓 프록시를 거쳐 OVH 채팅 성공. 스냅샷 `snapshots/03-p2`.
  - httpx2 전환은 보류: `http_client=httpx.Client` 계약이 깨지는 호환성 변경이라 사용자 승인이 필요하다.
  - `uv_build`는 `>=0.9.28,<0.10.0` 유지: uv 0.9.28이 uv_build 0.13.7 sdist를 풀지 못함(`malformed pax extension`).
    uv를 0.13으로 올릴 때 함께 올린다.
  - 하지 않음: 도구 호출 탐침(무료 한도를 소모하고 카탈로그 `supported_parameters`로 대부분 대체됨).
