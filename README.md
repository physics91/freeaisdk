# freeai

여러 플랫폼의 무료 LLM을 `chat()` 한 번으로 쓰는 Python SDK입니다. 대상은 OpenRouter, Groq, Gemini, Cloudflare, Z.ai 등 무료 등급이 있는 OpenAI 호환 플랫폼입니다.
지금 쓸 수 있는 무료 모델을 고르고, 한도를 미리 세어 429를 피합니다. 실패하면 원인에 맞게 그 모델이나 플랫폼을 쉬게 하고 다음 후보로 넘어갑니다.

- 동기 `FreeAI`, 비동기 `AsyncFreeAI`, OpenAI SDK 모양의 `freeai.compat.OpenAI`
- 로컬 OpenAI 호환 프록시 `freeai serve`: 기존 도구의 base URL만 바꾸면 됩니다
- 분, 시간, 일, 월 단위 요청과 토큰 한도. 응답 헤더로 카운터를 보정합니다
- SQLite로 여러 프로세스가 cooldown과 사용량을 공유합니다
- 프롬프트 학습 여부로 모델을 거르는 필터, 응답 거절 훅, OpenTelemetry 연동

## 설치

```bash
uv add /path/to/freeaisdk            # 로컬 경로 (PyPI에는 아직 올리지 않았습니다)
uv add /path/to/freeaisdk --extra otel   # OpenTelemetry 연동까지
```

Python 3.12 이상, 런타임 의존성은 `httpx` 하나입니다.

## 빠른 시작

```python
from freeai import chat

print(chat("파이썬 GIL을 한 문장으로 설명해줘"))   # 키가 설정된 무료 프리셋을 자동으로 묶습니다
```

```python
from freeai import FreeAI, OpenRouter, Groq, Cloudflare

ai = FreeAI(
    [OpenRouter(app_name="my-app"), Groq(), Cloudflare()],
    prefer=["*gpt-oss-120b*", "*gemma*"],   # 먼저 시도할 모델 (glob, "provider:id"도 가능)
    max_wait=30,                            # 모두 쉬는 중이면 최대 30초 대기
    deadline=60,                            # 호출 하나의 전체 시간 상한
)

response = ai.chat("안녕?", system="짧게 답해", temperature=0.3)
print(response.text, response.provider, response.model)
for attempt in response.attempts:           # 어떤 모델을 거쳐 왔는지
    print(attempt.model, attempt.ok, attempt.kind, attempt.cooldown)

for chunk in ai.stream("1부터 5까지 세어줘", first_token_timeout=15):
    print(chunk.text, end="", flush=True)
```

```python
import asyncio
from freeai import AsyncFreeAI

async def main():
    async with AsyncFreeAI.from_env() as ai:
        print((await ai.chat("hi", model="fast")).text)
        async for chunk in ai.stream("count to 3"):
            print(chunk.text, end="")

asyncio.run(main())
```

## 공급자

`FreeAI.from_env()`는 환경 변수가 설정된 프리셋 가운데 **무료 등급**만 켭니다. 체험 크레딧이나 유료 키는
`tiers=("free", "trial", "paid")`로 직접 허용해야 하고, 허용하면 `BillingWarning`이 뜹니다. 키 없는 익명
엔드포인트는 `include_keyless=True`일 때만 씁니다. 아래 표의 한도는 2026-11-06에 확인한 값입니다.
선언된 한도는 응답 헤더나 quota API로 계속 보정됩니다.

| 프리셋 | 환경 변수 | 등급 | 선언된 한도 | 프롬프트 학습 |
|---|---|---|---|---|
| `OpenRouter` | `OPENROUTER_API_KEY` | free | 계정 20 RPM, 일 한도는 `GET /key`에서 동기화 | 엔드포인트마다 다름. `allow_training=False`면 요청마다 거부 |
| `Groq` | `GROQ_API_KEY` | free | 모델별 30 RPM, 1000 RPD, 8K TPM, 200K TPD | 하지 않음 |
| `Gemini` | `GEMINI_API_KEY` | free | 모델별 Flash 5 RPM, 20 RPD, Flash-Lite 10 RPM, 500 RPD, 250K TPM (공식 수치 비공개라 제3자 자료, preview는 Flash로 가정). 유료 프로젝트는 `limits=[]`. 429의 `PerDay`이면 태평양 자정까지 쉼 | 함 (EEA, 스위스, 영국 서비스는 유료 등급 필요) |
| `Cloudflare` | `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID` | free | 모델별 300 RPM, 일 10,000 neurons (UTC 자정) | 하지 않음 |
| `ZAI` | `ZAI_API_KEY` | free | GLM Flash 모델, 동시 요청 1개 | 명시 없음 |
| `SambaNova` | `SAMBANOVA_API_KEY` | free | 모델별 20 RPM, 20 RPD, 200K TPD | 확인 못 함 |
| `NVIDIA` | `NVIDIA_API_KEY` | free | 계정 40 RPM (체험 엔드포인트) | 함 (기록되고 제품 개선에 쓰임) |
| `Kilo` | `KILO_API_KEY` 또는 익명 | free | 익명은 IP당 시간 200회 | 모델별 `mayTrainOnYourPrompts` |
| `Requesty` | `REQUESTY_API_KEY` | free | 계정 200 RPD, 가격 0 모델만 | 확인 못 함 |
| `OVH` | 없음 (익명) | free | IP와 모델당 2 RPM | 하지 않음 |
| `Cohere` | `COHERE_API_KEY` | trial | 월 1,000회, 모델별 20 RPM, 상업 이용 불가 | 확인 못 함 |
| `Cerebras` | `CEREBRAS_API_KEY` | trial | 5달러, 30일 체험 (카드 필요) | 확인 못 함 |
| `Mistral` | `MISTRAL_API_KEY` | paid | 종량제 | |

플랫폼 약관은 계정 여러 개로 한도를 늘리는 것을 금지하거나 권하지 않습니다. 이 SDK의 목적은 계정 하나씩을
여러 플랫폼에 걸쳐 쓰는 것입니다.

직접 붙이려면 `Provider(name, base_url, api_key=..., models=[...], limits=[rpm(30), rpd(1000)])`처럼 만듭니다.
`models`를 생략하면 `GET /models` 전체를 쓰는데, 무료 플랜에서 막힌 모델이 섞일 수 있습니다.

공급자가 직접 만드는 httpx 클라이언트는 연결 1000개, keep-alive 100개까지 엽니다
(`freeai.providers.CONNECTION_LIMITS`, OpenAI 공식 SDK와 같은 값). 풀 크기나 프록시를 바꾸려면 직접 만든
클라이언트를 `http_client=`(동기)나 `async_http_client=`(비동기)로 넘깁니다. 넘긴 클라이언트는 SDK가 닫지 않습니다.

## 동작 방식

1. **후보 고르기**: 카탈로그(무료 모델만, 1시간 캐시)에서 요청 조건을 만족하는 모델을 고릅니다.
   `tools`, `response_format`, 이미지 입력이 있으면 그 기능을 지원하는 모델만 남습니다. 라우터 모델
   (`openrouter/free`)은 맨 뒤에 둡니다.
2. **예약**: 후보마다 cooldown, 한도 창, 동시 요청 슬롯을 확인하고 지금 보낼 수 있는 첫 모델을 예약합니다.
   한도는 분과 시간 단위가 GCRA 방식의 이동 창이고, 일과 월 단위는 공급자 시간대의 경계에서 리셋되는 고정
   창입니다. 응답 헤더가 다르다고 알려 주면 그쪽을 따릅니다(`RateSnapshot.rolling`). Groq의 일일 요청 한도는
   이동 창으로 읽습니다. 실측해 보니 Groq는 직전 요청 뒤로 86.4초가 지날 때마다 하나씩 돌려주고, 요청이 오면 남은
   자투리 시간을 버립니다. 그래서 86.4초보다 촘촘히 보내는 동안에는 다시 차지 않습니다. SDK는 응답마다 헤더로
   수를 다시 맞추고, 다 쓰면 마지막 헤더에서 86.4초(여유 1초 더) 뒤에 보냅니다. Groq가 하나를 돌려준 뒤입니다.
   OVHcloud의 분당 한도는 `ratelimit-reset` 초 뒤에 한꺼번에 리셋되는 고정 창입니다. 고정 창은 서버가 알려 준 리셋 시각에서 1초 더 기다린 뒤 다시 씁니다.
   이동 창이 시간이 지나 되찾은 몫은 1초(`freeai.limits.EDGE_MARGIN`) 늦게 씁니다. 요청이 플랫폼에 닿기까지 걸리는
   시간이 매번 달라서, 경계에 맞춰 보낸 요청이 몇 ms 일찍 도착해 429를 받기 때문입니다. 창이 가득 찬 상태에서
   한꺼번에 보내는 버스트는 늦추지 않고, 계속 한도까지 쓰는 경우의 처리량도 그대로입니다. 창당 1회 한도만
   1초씩 손해를 봅니다(분당 1회면 약 1.6%).
3. **응답 헤더 반영**: `x-ratelimit-limit/remaining/reset-{requests,tokens}[-day|-minute]` 헤더와 OpenRouter
   `GET /key`로 로컬 카운터를 갱신합니다. 보냈지만 아직 응답 헤더가 오지 않은 요청은 플랫폼이 세지 않았으므로
   그 예약분을 서버 값에 더합니다. 예약분은 저장소에 두므로 `SQLiteStore`를 함께 쓰는 다른 프로세스의 것도
   들어갑니다. 5분(공급자 `timeout`의 두 배가 더 길면 그만큼) 안에 정리되지 않은 예약분은 죽은 프로세스의 것으로
   보고 버립니다. 스트림은 헤더가 도착하는 순간 반영해서, 스트림이 끝날 때 오래된 헤더로 그사이 시작된 요청을
   지우지 않습니다. 토큰은 추정치로 먼저 잡고 헤더나 `usage`로 정산합니다. 동시에 보낸 요청의 응답은 플랫폼이
   센 순서와 다르게 처리될 수 있습니다. 플랫폼은 요청 수를 돌려주지 않으므로, 이미 반영한 것보다 적게 센 헤더는
   늦게 온 것으로 보고 버립니다. 토큰은 쓰지 않은 몫이 돌아와 실제로 줄 수 있으므로, 이미 끝난 창의 헤더만
   버립니다.
   쓸 모델이 없어 기다리던 요청은 다른 요청의 헤더가 도착하거나 요청이 끝나 예약이 풀리면, 계산해 둔 대기 시간을
   다 채우지 않고 바로 다시 확인합니다. 같은 모델 후보로 기다리는 요청끼리 묶어, 변경 하나에 묶음마다 먼저 기다린
   요청 하나만 깨우므로 한꺼번에 몰려가지 않습니다. `SQLiteStore`를 함께 쓰는 다른 프로세스의 변경은 0.25초마다
   확인합니다. 직접 만든 저장소도 다른 프로세스가 바꿀 때마다 달라지는 정수를 돌려주는 `version()`을 두면 같은
   방식으로 깨어나고, 없으면 계산한 시간까지 잡니다. `max_wait`는 실제로 기다린 시간으로 셉니다. `sleep`을 바꿔
   넘기면(테스트용) 이 기능은 꺼집니다.
4. **실패 분류**: 실패마다 종류(`FailureKind`)와 범위(`Scope`)를 정합니다.

   | 실패 | 예 | 쉬는 범위 |
   |---|---|---|
   | 분당 한도 | 429, 상위 공급자 429 | 그 모델 |
   | 계정 한도 | OpenRouter `free-models-per-min`, Z.ai `1313` | 플랫폼 전체 |
   | 일일, 월간 소진 | `free-models-per-day`, Gemini `PerDay`, Cloudflare `3036`, Z.ai `1308` | 리셋 시각까지 |
   | 장애 | 408, 5xx, 타임아웃, HTTP 200 안의 `error` | 그 모델 (지수 백오프) |
   | 빈 응답, 모델 삭제, 권한 없음 | 내용 없음, 404, 데이터 정책 404 | 그 모델 |
   | 키 오류, 크레딧 부족 | 401, 402, Gemini 프로젝트 인증 403 | 플랫폼 전체 |
   | 잘못된 요청, 문맥 초과 | 400, 413, 422, 요청 하나가 분당 한도를 넘는 Groq 429(`Request too large`) | 이번 요청에서만 제외. 같은 플랫폼이 두 번 거절하면 그 플랫폼 중단 |
   | 출력 예산 소진 | `finish_reason=length`인데 내용 없음 (추론 모델) | 재시도하지 않고 `OutputLimitError` |

   `Retry-After`, 리셋 헤더, Gemini `RetryInfo`, 본문의 "retry in 40s" 같은 힌트가 있으면 그 시간을 따릅니다.
5. **복구**: cooldown이 끝난 모델이나 플랫폼에는 요청 하나만 먼저 보내고(half-open), 결과가 나올 때까지 다른 요청은
   기다립니다. 5xx와 타임아웃이 몰리면 재시도 예산(`RetryBudget`)이 바닥나 장애 중에 모든 무료 엔드포인트를
   훑지 않습니다.
6. **스트리밍**: 첫 출력 전 실패만 다른 모델로 넘깁니다. 출력이 나간 뒤 실패하면 `StreamInterrupted`에 받은
   텍스트(`partial_text`)를 담아 올립니다.

모든 후보가 실패하거나 `max_wait`, `deadline` 안에 쓸 모델이 없으면 `ExhaustedError`가 납니다. 여기에는
`attempts`와 `next_available_in`이 들어 있습니다.

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `strategy` | `"priority"` | `"round_robin"`, `"random"`, `"health"`(성공률과 지연의 지수 평균으로 두 후보 중 나은 쪽 선택) |
| `prefer`, `exclude` | `()`, `DEFAULT_EXCLUDE` | 모델 패턴. 기본 제외는 `*guard*`, `*content-safety*`, `*whisper*`, `*tts*`, `*embed*` |
| `model=` (호출별) | 없음 | id, glob, `provider:id`, 목록, 별칭 `fast`, `smart`, `coding`, `auto` |
| `aliases` | 내장 별칭 | `{"mine": ["*qwen*"]}`. 별칭은 순서만 바꾸고 다른 모델을 막지 않습니다 |
| `max_attempts`, `max_wait` | `8`, `30.0` | 호출당 최대 시도 수, 쉬는 모델을 기다릴 최대 초 |
| `deadline` | `None` | 호출 전체 시간 상한(초). 각 HTTP 읽기 타임아웃도 남은 시간으로 줄입니다 |
| `first_token_timeout` | `None` | 스트림이 이 시간 안에 이벤트를 하나도 보내지 않으면 다른 모델로 넘깁니다. 출력이 시작된 뒤에도 읽기 타임아웃으로 남으므로, 그보다 길게 멈추면 `StreamInterrupted`가 납니다 |
| `allow_training` | `True` | `False`면 학습하지 않는다고 밝힌 모델만 씁니다. OpenRouter에는 `provider.data_collection="deny"`를 보냅니다 |
| `reject_response` | `None` | `fn(ChatResponse) -> bool \| str`. 참이면 그 답을 버리고 다음 모델로 넘어갑니다 (스트림 제외) |
| `store` | `MemoryStore()` | `SQLiteStore("state.db")`로 프로세스 사이에 cooldown과 사용량 공유 |
| `retry_budget` | `True` | `RetryBudget(...)`으로 조정하거나 `False`로 끕니다 |
| `cooldowns` | `CooldownPolicy()` | 종류별 기본 cooldown, 최대 백오프 |
| `on_attempt` | `None` | 시도마다 호출되는 콜백 (`Attempt`) |
| `clock` | `time.time` | 벽시계 초. 상태를 epoch로 저장하므로 단조 시계를 넘기면 안 됩니다 |

`ai.models(require=["tools"])`는 시도 순서, `ai.status()`는 남은 cooldown, `ai.health()`는 모델별 성공률과 지연을 보여줍니다.

## OpenAI SDK 모양으로 쓰기

```python
from freeai.compat import OpenAI

client = OpenAI()   # FreeAI.from_env() 사용, OpenAI(FreeAI(...))도 가능
reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
print(reply["choices"][0]["message"]["content"], reply["x_freeai"])
```

응답은 OpenAI 형식의 dict이고 `x_freeai`에 실제로 답한 공급자와 모델이 들어 있습니다. `AsyncOpenAI`도 있습니다.

## 로컬 프록시

```bash
freeai serve                              # http://127.0.0.1:8787/v1
freeai --keyless --no-training serve      # 익명 엔드포인트 포함, 학습하는 모델 제외
FREEAI_PROXY_TOKEN=... freeai serve --host 0.0.0.0
```

`/v1/chat/completions`(스트림 포함)와 `/v1/models`, `/health`를 제공합니다. 응답 헤더 `X-FreeAI-Provider`,
`X-FreeAI-Model`이 실제로 답한 모델을 알려 줍니다. 모든 모델이 쉬는 중이면 `429`와 `Retry-After`를 돌려줍니다.
OpenAI SDK 클라이언트는 `max_retries=0`으로 두세요. 재시도와 전환은 프록시가 이미 합니다.

보안: 프록시는 시작할 때 읽은 모든 API 키의 한도를 대신 씁니다. 그래서 기본 바인딩은 루프백이고, 다른 주소에는
bearer 토큰이 없으면 시작하지 않습니다. 토큰은 상수 시간으로 비교합니다. 오류 본문에서 키 문자열을 가리고, 요청
본문은 16 MiB로 제한합니다. 클라이언트는 공급자 키를 보내지 않습니다.

## 명령줄

```bash
freeai models --require tools        # 시도 순서대로 후보 출력
freeai chat "질문" --model smart --stream
freeai --store ~/.cache/freeai.db chat "..."   # 다른 프로세스와 상태 공유
```

## 관측

```python
from freeai.otel import instrument
instrument(ai)   # 시도마다 GenAI 규약 span 하나 (gen_ai.provider.name, gen_ai.request.model, error.type ...)
```

`opentelemetry-api`가 필요합니다(`freeai[otel]`). GenAI 시맨틱 규약이 아직 Development 상태라 실험 기능입니다.
프롬프트와 응답 내용은 기록하지 않습니다. 로그는 `logging.getLogger("freeai")`로 나가며 기본 핸들러는 `NullHandler`입니다.

## 한계

- health 통계, 재시도 예산, half-open 탐침, `max_concurrency` 슬롯은 프로세스 안에서만 유지됩니다.
  `SQLiteStore`는 cooldown, 한도 카운터, 헤더를 받기 전 요청의 예약분만 공유합니다. 이 저장소는 로컬 디스크의 WAL 모드가 전제라 NFS에서는 쓰지 마세요.
  `FreeAI.close()`는 넘겨받은 `store`를 닫지 않으므로, 저장소의 수명은 호출한 쪽이 관리합니다.
- 별칭(`fast`, `smart`, `coding`)은 모델 id 이름 규칙에 기댄 휴리스틱입니다.
- 상태 키는 공급자 이름과 API 키 SHA-256 앞 12자로 계정을 구분합니다. 키 자체는 저장하지 않습니다.
- 선언된 무료 한도는 플랫폼이 수시로 바꿉니다. 각 `Limit`에 `source`와 `last_verified`를 남겨 두었습니다.
- 키가 필요한 플랫폼 가운데 Groq만 키로 실측했습니다(채팅, 스트림, 헤더, `Request too large` 429 본문).
  나머지(OpenRouter 채팅, Gemini, Cloudflare, Z.ai, SambaNova, NVIDIA, Requesty, Cohere)의 실제 채팅과 429 본문 모양은
  키가 없어 실측하지 못했습니다. 분류 규칙은 공식 문서와 공개된 응답 예시에 기반합니다.
- 스트리밍이 아닌 `chat()`은 응답이 끝나야 헤더가 옵니다. 그동안 플랫폼이 이미 센 요청도 예약분으로 한 번 더
  세므로 보수적으로 동작합니다.
- 플랫폼이 요청을 받는 순간 센다고 가정합니다. OVHcloud와 Groq에서 실측으로 확인했습니다
  (`examples/chat_app/probe_headers.py`). 다른 플랫폼은 같은 스크립트로 요청 세 번에 확인할 수 있습니다.
- Groq는 요청을 받는 순간 채팅 템플릿을 포함한 프롬프트와 `max_tokens`를 분당 토큰에서 미리 뺍니다. 템플릿 몫은
  프리셋이 `prompt_overhead_tokens = 80`으로 더합니다(실측 15~74토큰). `max_tokens`가 없으면 Groq가 잡는
  출력 몫이 2~961토큰으로 매번 달라 미리 알 수 없습니다(gpt-oss는 주로 최근 응답 하나의 길이). SDK는 512로
  어림하므로, Groq에서 정확히 맞추려면 `max_tokens`를 주세요. qwen은 분당 출력 토큰 1000 한도가 따로 있어,
  `max_tokens` 없는 요청 12번 중 1번이 `Request too large` 429를 받았습니다. `max_tokens` 1500 요청은 받아들여져서,
  무엇이 이 오류를 부르는지는 모릅니다. Groq가 요청 자체가 크다고 알리므로 SDK는 그 요청에서만 모델을 뺍니다.
- Groq 일일 요청은 86.4초보다 촘촘히 보내는 동안 다시 차지 않습니다(실측). 하루 1000건은 사실상 한 번 쓰면
  쉬는 시간에만 돌아오는 몫으로 보세요. 1000건을 실제로 다 쓴 상태는 확인하지 못했습니다. 회복 방식만
  실측했고(`reports/load-test/groq-daily-refill-real.jsonl`), 다 쓴 뒤 86.4초 간격으로 보내는 동작은 단위 테스트로
  확인했습니다.
- OVHcloud의 분당 한도를 기다리면 최대 약 61초가 걸립니다. `max_wait`가 그보다 짧으면 기다리지 않고 다른
  모델로 넘어갑니다.
- Z.ai의 동시 요청 1개 제한, `next_flush_time` 시간대(UTC+8), NVIDIA 무료 엔드포인트 목록은 공식 확인이 안 된 값입니다.

## 개발

```bash
uv sync
uv run pytest                                    # 모의 HTTP, 프록시는 실제 소켓
uv run --with opentelemetry-sdk pytest           # OTel span 테스트 포함
FREEAI_LIVE=1 uv run pytest tests/test_live.py   # 실제 네트워크 (OVH 익명 채팅 포함, 키가 있으면 OpenRouter 채팅)
uv run --with ruff ruff check src tests && uv run --with mypy mypy
uv build
```

라이선스: MIT
