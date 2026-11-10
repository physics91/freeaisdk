# freeai

> **For AI agents**  
> Install `free-llm-sdk` (not `freeai` / `freeaisdk`). Import `freeai`.  
> Fast path: `from freeai import chat; print(chat("..."))` — no API key required.  
> Or `FreeAI.from_env()`, or `freeai serve` → `http://127.0.0.1:8787/v1` (`max_retries=0`).  
> Details: [`llms.txt`](./llms.txt) · [`docs/agent/guide.md`](./docs/agent/guide.md) · [consumer skill](./.agents/skills/freeai/SKILL.md)

여러 플랫폼의 무료 LLM을 `chat()` 한 번으로 쓰는 Python SDK입니다. 대상은 OpenRouter, Groq, Gemini, Cloudflare, Z.ai 등 무료 등급이 있는 OpenAI 호환 플랫폼입니다.
지금 쓸 수 있는 무료 모델을 고르고, 한도를 미리 세어 429를 피합니다. 실패하면 원인에 맞게 그 모델이나 플랫폼을 쉬게 하고 다음 후보로 넘어갑니다.

- 동기 `FreeAI`, 비동기 `AsyncFreeAI`, OpenAI SDK 모양의 `freeai.compat.OpenAI`
- 로컬 OpenAI 호환 프록시 `freeai serve`: 기존 도구의 base URL만 바꾸면 됩니다
- 분, 시간, 일, 월 단위 요청과 토큰 한도. 응답 헤더로 카운터를 보정합니다
- 다시 차는 몫부터 쓰는 순서(기본 전략 `"quota"`), 쿼터가 찰 때까지 기다리는 `batch()`, 추정 한도를 스스로 맞추는 `adapt_limits`
- SQLite로 여러 프로세스가 cooldown과 사용량을 공유합니다
- 프롬프트 학습 여부로 모델을 거르는 필터, 응답 거절 훅, OpenTelemetry 연동

## 설치

PyPI 배포 이름은 `free-llm-sdk`입니다(`freeai` / `freeaisdk`는 유사도 검사에 걸림). import와 CLI는 `freeai`입니다.

```bash
uv add free-llm-sdk
uv add "free-llm-sdk[otel]"       # OpenTelemetry 연동까지
pip install free-llm-sdk

# 로컬 경로
uv add /path/to/freeaisdk
uv add /path/to/freeaisdk --extra otel
```

Python 3.12 이상, 런타임 의존성은 `httpx` 하나입니다.

## 빠른 시작

```python
from freeai import chat

print(chat("파이썬 GIL을 한 문장으로 설명해줘"))   # 키가 설정된 무료 프리셋과 익명 OVHcloud를 자동으로 묶습니다
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
`tiers=("free", "trial", "paid")`로 직접 허용해야 하고, 허용하면 `BillingWarning`이 뜹니다. 아래 표의 한도는
2026-11-06에 확인한 값입니다. 선언된 한도는 응답 헤더나 quota API로 계속 보정됩니다.

키 없는 익명 엔드포인트는 계정이 없는 플랫폼으로 프롬프트를 보냅니다. 그래서 기본으로는 프롬프트를 학습에 쓰지
않는다고 밝힌 OVHcloud만 씁니다. 키가 하나도 없어도 `from_env()`와 `chat()`이 OVHcloud로 동작합니다.
`include_keyless=True`면 Kilo도 키 없이 쓰고(무료 모델이 프롬프트를 학습에 쓸 수 있음), `include_keyless=False`면
익명 엔드포인트를 쓰지 않고 내 키만 씁니다.

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
| `OVH` | 없음 (익명) | free | IP와 모델당 2 RPM, 모델 9개 | 하지 않음 |
| `Cohere` | `COHERE_API_KEY` | trial | 월 1,000회, 모델별 20 RPM, 상업 이용 불가 | 확인 못 함 |
| `Cerebras` | `CEREBRAS_API_KEY` | trial | 5달러, 30일 체험 (카드 필요) | 확인 못 함 |
| `Mistral` | `MISTRAL_API_KEY` | paid | 종량제 | |

플랫폼 약관은 계정 여러 개로 한도를 늘리는 것을 금지하거나 권하지 않습니다. 이 SDK의 목적은 계정 하나씩을
여러 플랫폼에 걸쳐 쓰는 것입니다.

직접 붙이려면 `Provider(name, base_url, api_key=..., models=[...], limits=[rpm(30), rpd(1000)])`처럼 만듭니다.
같은 창을 두 번 선언하면 나중 것이 적용됩니다. 예를 들어 `NVIDIA(rpm=20)`은 프리셋의 계정 40 RPM을 20으로 바꿉니다.
`models`를 생략하면 `GET /models` 전체를 쓰는데, 무료 플랜에서 막힌 모델이 섞일 수 있습니다.

공급자가 직접 만드는 httpx 클라이언트는 연결 1000개, keep-alive 100개까지 엽니다
(`freeai.providers.CONNECTION_LIMITS`, OpenAI 공식 SDK와 같은 값). 풀 크기나 프록시를 바꾸려면 직접 만든
클라이언트를 `http_client=`(동기)나 `async_http_client=`(비동기)로 넘깁니다. 넘긴 클라이언트는 SDK가 닫지 않습니다.

## 동작 방식

1. **후보 고르기**: 카탈로그(무료 모델만, 1시간 캐시)에서 요청 조건을 만족하는 모델을 고릅니다.
   `tools`, `tool_choice`, `response_format`, `reasoning`, 이미지 입력이 있으면 그 기능을 지원하는 모델만 남습니다.
   꺼진 값(빈 `tools`, `tool_choice="none"`, `response_format={"type": "text"}`, `reasoning=False`)은 후보를
   거르지 않습니다. 라우터 모델(`openrouter/free`)은 맨 뒤에 둡니다.
2. **예약**: 후보마다 cooldown, 한도 창, 동시 요청 슬롯을 확인하고, 지금 보낼 수 있는 모델 가운데 쓴 몫이 가장 빨리
   돌아오는 것을 예약합니다(기본 전략 `"quota"`, 아래 "무료 쿼터를 끝까지 쓰기"). `strategy="priority"`면 순서상 첫 모델입니다.
   한도는 분과 시간 단위가 GCRA 방식의 이동 창이고, 일과 월 단위는 공급자 시간대의 경계에서 리셋되는 고정
   창입니다. 응답 헤더가 다르다고 알려 주면 그쪽을 따릅니다(`RateSnapshot.rolling`). Groq의 일일 요청 한도는
   이동 창으로 읽습니다. 실측해 보니 Groq는 직전 요청 뒤로 86.4초가 지날 때마다 하나씩 돌려주고, 요청이 오면 남은
   자투리 시간을 버립니다. 그래서 86.4초보다 촘촘히 보내는 동안에는 다시 차지 않습니다. SDK는 응답마다 헤더로
   수를 다시 맞추고, 다 쓰면 마지막 헤더에서 86.4초(여유 1초 더) 뒤에 보냅니다. Groq가 하나를 돌려준 뒤입니다.
   Groq의 분당 요청 한도도 같은 방식이라 프리셋이 `rpm(30, whole=True)`로 선언합니다. 마지막으로 받아들인 요청 뒤로
   온전한 2초마다 하나를 돌려주고 자투리는 버리며, 거절한 요청은 세지 않습니다. 그래서 버스트 중에는 다시 차지 않고,
   다 쓴 뒤에는 2.1초(여유 `freeai.limits.WHOLE_MARGIN` 0.1초 포함) 간격으로 보냅니다. 이 창은 응답 헤더로 알 수 없어
   RPM 429 본문의 `Limit`과 `Used`로 수를 다시 맞춥니다.
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
   늦게 온 것으로 보고 버립니다. 토큰은 쓰지 않은 몫이 돌아와 실제로 줄 수 있으므로, 이미 끝난 창의 헤더와,
   이미 반영한 헤더의 요청보다 먼저 보낸 요청의 헤더만 버립니다. 스트리밍이 아닌 응답의 헤더는 본문과 함께
   도착하지만 값은 요청을 받은 순간의 것이라(Groq 실측), 나중에 보낸 요청이 쓴 토큰을 모르기 때문입니다.
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
   | 차단, 답 거절 | 403 검열 차단, `reject_response`가 버린 답 | 이번 요청에서만 제외 (cooldown 없음) |
   | 출력 예산 소진 | `finish_reason=length`인데 답(텍스트, tool call)이 없음 (추론 모델, 스트림 포함) | 재시도하지 않고 `OutputLimitError` |

   `Retry-After`, 리셋 헤더, Gemini `RetryInfo`, 본문의 "retry in 40s" 같은 힌트가 있으면 그 시간을 따릅니다.
5. **복구**: cooldown이 끝난 모델이나 플랫폼에는 요청 하나만 먼저 보내고(half-open), 결과가 나올 때까지 다른 요청은
   기다립니다. 5xx와 타임아웃이 몰리면 재시도 예산(`RetryBudget`)이 바닥나 장애 중에 모든 무료 엔드포인트를
   훑지 않습니다.
6. **스트리밍**: 첫 출력 전 실패만 다른 모델로 넘깁니다. 출력이 나간 뒤 실패하면 `StreamInterrupted`에 받은
   텍스트(`partial_text`)를 담아 올립니다. 출력은 텍스트, 추론, tool call이고, 첫 출력 전에 오는 `role`만 담은
   청크와 공백뿐인 청크는 건너뜁니다.
   스트림이 오류 없이 끝났는데 답(텍스트나 tool call)이 없으면 `chat()`과 같이 처리합니다. `finish_reason=length`면
   `OutputLimitError`를 올리고, 아니면 다른 모델로 넘깁니다(`retry_on_empty=False`면 그대로 끝냅니다). 추론만 보내고
   끝난 스트림도 여기에 들어가므로, 호출자는 앞 모델의 추론 청크를 받은 뒤 다음 모델의 청크를 받을 수 있습니다.
   청크마다 들어 있는 `provider`와 `model`로 구분하세요.

모든 후보가 실패하거나 `max_wait`, `deadline` 안에 쓸 모델이 없으면 `ExhaustedError`가 납니다. 여기에는
`attempts`와 `next_available_in`이 들어 있습니다.

## 무료 쿼터를 끝까지 쓰기

`strategy="quota"`와 `adapt_limits=True`는 기본값입니다. 이전 동작이 필요하면 `strategy="priority"`,
`adapt_limits=False`를 주세요. `batch()`와 `audit`은 부를 때만 동작합니다. 그 밖에 바뀐 동작(저장된 한도 크기,
진행 중 요청의 이중 계산, 같은 창의 중복 선언 등)은 `CHANGELOG.md`에 있습니다.

```python
ai = FreeAI.from_env()                       # 키가 있는 무료 프리셋에, 분 단위로 다시 차는 익명 OVHcloud까지
answers = ai.batch(prompts, concurrency=8)   # 쿼터가 다시 찰 때까지 기다리며 전부 처리, 입력 순서대로
```

얻는 것은 아껴 둔 일일 몫만큼입니다. 수요가 분 단위로 다시 차는 몫 아래일 때는 어느 전략이든 전부 처리하고, 그
위로 몰릴 때 일일 몫이 남아 있느냐가 차이를 만듭니다. 평소 분당 30건, 마지막에 분당 90건으로 몰리는 모의 하루
(분당 40회 플랫폼 하나, 하루 200회 플랫폼 하나)에서 `priority`는 1,800건 중 1,338건을, 기본 전략은 1,538건을
처리했습니다. 늘어난 200건이 일일 몫 전부입니다.

**`strategy="quota"`: 다시 차는 몫부터 씁니다.** 지금 보낼 수 있는 후보 가운데, 이 요청이 쓴 몫을 플랫폼이 돌려주는
데 걸리는 시간이 가장 짧은 모델로 보냅니다. 분당 40회 한도는 요청 하나에 1.5초, 하루 1000회는 86.4초, 새 날의
하루 50회는 1728초입니다. 그래서 분 단위로 다시 차는 몫(OVHcloud, NVIDIA, 익명 Kilo)을 먼저 쓰고, 일일 몫
(OpenRouter, Groq, Gemini)은 넘치는 요청만 받습니다.

- 고정 창은 남은 양과 리셋까지 남은 시간으로 계산합니다. 마지막 한 건은 남은 시간 전부만큼 비싸고, 리셋 직전에
  남아 있는 몫은 어차피 사라지므로 먼저 씁니다.
- 토큰 한도도 같은 식이라 큰 요청은 토큰 여유가 큰 플랫폼으로 갑니다.
- `prefer`, 호출의 `model`, 라우터 모델을 맨 뒤에 두는 규칙이 먼저입니다. 쿼터는 같은 묶음 안에서만 순서를 정합니다.
- 무료 등급을 체험 크레딧과 유료 키보다 먼저 씁니다.
- 크기를 셀 수 없는 한도는 창만 선언해 두었습니다(`Provider.uncounted`: Cloudflare 일일 neurons, Z.ai 사용 한도,
  키를 쓴 Kilo). 수치는 지어내지 않고, 그 창이 끝날 때까지 아껴 쓰는 쪽으로만 정렬합니다.
- 등급을 밝히지 않은 `Provider`(직접 붙인 서버 등, `tier="unknown"`)는 무료 등급 뒤에 씁니다. 한도를 선언하지 않으면
  가장 싼 후보로 보여 모든 요청을 가져가는데, 유료 엔드포인트일 수도 있기 때문입니다. 무료 후보가 모두 막혔을 때는
  바로 쓰므로 처리량은 줄지 않습니다. 먼저 쓰려면 `tier="free"`를 주거나 `prefer`에 넣으세요.
- 대가가 있습니다. 분당 몫만 있는 플랫폼은 느리거나(익명 OVHcloud) 프롬프트를 학습에 씁니다(NVIDIA, Kilo).
  Groq와 OVHcloud로 요청 8번을 보낸 실측(2026-11-10)에서 `priority`는 4건 모두 Groq가 0.3~0.6초에,
  `quota`는 4건 모두 OVHcloud가 0.6~3.5초에 답했습니다. 표본이 작은 수치이고 꼬리는 더 깁니다. 익명 OVHcloud에
  94번 시도한 부하 측정(2026-11-06, `reports/샘플 채팅앱 부하 측정.md`)에서는 첫 토큰 p95가 7~16초, 답 전체 p95가
  24~29초였고, 지금 OVHcloud의 첫 후보인 `Qwen3.8-27B`가 20초 안에 첫 토큰을 보내지 않은 적도 있습니다.
  속도가 중요하면 `prefer=["groq:*"]`처럼 빠른 플랫폼을 앞에 두거나, `deadline`을 걸거나, `include_keyless=False`로
  익명 엔드포인트를 빼세요. 학습이 문제면 `allow_training=False`를 함께 쓰세요.

**`batch()`와 호출별 `max_wait`: 쿼터가 찰 때까지 기다립니다.** `chat()`과 `stream()`은 `max_wait`를 호출마다 받습니다
(`math.inf`면 끝까지 기다림). `batch(prompts, concurrency=8)`는 여러 건을 동시에 돌리고, 결과를 입력 순서대로
돌려줍니다. 각 항목은 응답이거나 그 요청이 끝난 `FreeAIError`입니다. 한 건이 실패해도 나머지는 계속합니다.
`batch()`의 `max_wait` 기본값은 제한 없음이라, 남은 것이 일일 한도뿐이면 리셋까지 잡니다. 일찍 포기하려면
`max_wait`나 `deadline`을 주세요(둘 다 건별). 그 밖의 예외와 Ctrl-C는 배치를 멈추고 기다리던 요청을 깨웁니다.
분당 2회인 실제 OVHcloud 모델 하나에 4건을 넣은 실측에서 2건은 40.5초, 2건은 87.9초에 끝났고 업스트림 429는
없었습니다. `max_wait=0`이면 2건만 답하고 2건은 `ExhaustedError`였습니다.

**`adapt_limits`: 추정으로 선언한 일일 한도를 실제에 맞춥니다(실험 기능, 기본으로 켜짐).** 공개되지 않아 제3자 수치로
선언한 한도(`Limit.estimate`) 가운데 일, 월 단위 요청 수가 대상입니다. 지금 프리셋에서는 Gemini의 일일 요청 수
(Flash 20, Flash-Lite 500)입니다. 헤더, 429 본문, quota API가 알려 주는 창은 건드리지 않습니다.

- 그 창만 요청을 막고 있으면 요청 하나를 더 보냅니다. 받아들여지면 한도를 25%(최소 1) 올리고, 새 한도를 넘는 다음
  요청이 다시 탐침이 됩니다. 그래서 실제 한도가 훨씬 커도 하루면 찾습니다. 거절되면 한도는 그대로 두고 다음 탐침을
  하루 뒤로 미루며, 연달아 거절될 때마다 두 배로 늘립니다(최대 7일).
- 창에 여유가 있다고 봤는데 플랫폼이 일일 한도 소진을 알리면, 그날 받아들여진 수로 한도를 내립니다.
- 그날 실패한 요청이 하나라도 있으면 탐침도, 내리기도 하지 않습니다. 실패한 요청은 로컬에서는 센 채로 두는데
  플랫폼은 세지 않았을 수 있어서, 그날의 집계가 증거가 되지 못하기 때문입니다. 다음 날 다시 판단합니다.
- 배운 값은 저장소의 별도 키(`qa:`)에 두므로 옵션을 끄면 선언값으로 돌아가고, `SQLiteStore`면 재시작 뒤에도 남습니다.
  선언값이 바뀌면 배운 값은 버립니다.
- 모의 서버에 열흘 동안 하루 60건씩 넣은 결과입니다. 선언 20, 실제 50이면 처리 200건이 500건이 되고 429는
  4번이었습니다. 선언 20, 실제 12면 429가 10번에서 5번으로 줄었습니다. 선언이 정확하면(20) 탐침 값으로 429가
  4번 생깁니다.
- 분, 시간 단위 한도는 추정이어도 적응하지 않습니다(NVIDIA의 40 RPM, Gemini의 RPM). 이동 창 너머로 보낸 요청이
  받아들여지는지는 플랫폼이 버스트를 허용하는 버킷인지 슬라이딩 창인지에 달려 있는데, 그것이 바로 모르는 정보입니다.
  두 방식을 모두 흉내 낸 모의 실행에서, 시도한 규칙마다 한쪽에서 정확한 한도를 잘못 올리거나 내렸고 나쁜 경우
  한 시간에 429가 300번 넘게 났습니다.

**`freeai audit`: 고정 모델 목록이 낡았는지 봅니다.** 프리셋이 고정한 모델 목록을 플랫폼의 `GET /models`와 비교해
사라진 모델과 목록에 없는 모델을 보여 줍니다(`ai.audit()`). 읽기만 하고 라우팅은 바꾸지 않습니다. 카탈로그만으로는
무료인지, 쓸 만한지 알 수 없기 때문입니다. 2026-11-10에 돌려 보니 OVHcloud에 고정하지 않은 채팅 모델 3개가
있었고, 익명 요청 하나씩으로 확인한 뒤 프리셋에 넣었습니다(모델 6개에서 9개, 분당 12회에서 18회). Groq의
`allam-2-7b`는 영어 질문에 아랍어로 답해서 넣지 않았습니다. 사라진 고정 모델이 있으면 종료 코드가 1입니다.
GitHub Actions 워크플로 `catalog-audit`가 매일(그리고 `workflow_dispatch`로) 같은 검사를 돌립니다. `missing`이거나
(`--fail-on-catalog-error`) 카탈로그를 못 읽으면 실패하고, `unpinned`는 Job Summary와 notice로만 남겨 사람이
확인한 뒤 핀합니다. 저장소 secrets에 넣은 무료 등급 키만 비교에 쓰이며, 키가 없으면 그 프리셋은 건너뜁니다
(OVHcloud는 익명이라 항상 포함). 키가 하나도 없으면 커버리지 경고가 납니다.

## 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `strategy` | `"quota"` | 쓴 몫이 가장 빨리 돌아오는 모델부터 씁니다. `"priority"`(공급자와 모델을 나열한 순서), `"round_robin"`, `"random"`, `"health"`(성공률과 지연의 지수 평균으로 두 후보 중 나은 쪽 선택) |
| `prefer`, `exclude` | `()`, `DEFAULT_EXCLUDE` | 모델 패턴. 기본 제외는 `*guard*`, `*content-safety*`, `*whisper*`, `*tts*`, `*orpheus*`, `*embed*`, `*bge*`, `*stable-diffusion*` |
| `model=` (호출별) | 없음 | id, glob, `provider:id`, 목록, 별칭 `fast`, `smart`, `coding`, `auto` |
| `aliases` | 내장 별칭 | `{"mine": ["*qwen*"]}`. 별칭은 순서만 바꾸고 다른 모델을 막지 않습니다 |
| `max_attempts`, `max_wait` | `8`, `30.0` | 호출당 최대 시도 수, 쉬는 모델을 기다릴 최대 초(`math.inf`면 끝까지). 둘 다 호출별로도 줄 수 있습니다 |
| `adapt_limits` | `True` | 추정으로 선언된 일, 월 단위 요청 한도(`Limit.estimate`)를 탐침과 429로 실제 값에 맞춥니다. 실험 기능. `False`면 선언값 그대로 |
| `deadline` | `None` | 호출 전체 시간 상한(초). 각 HTTP 읽기 타임아웃도 남은 시간으로 줄입니다 |
| `first_token_timeout` | `None` | 스트림이 이 시간 안에 실제 출력(텍스트, 추론, tool call)을 보내지 않으면 다른 모델로 넘깁니다. `role`만 담은 첫 이벤트, 빈 `choices`, keepalive 주석은 출력으로 치지 않습니다. 출력이 시작된 뒤에도 읽기 타임아웃으로 남으므로, 그보다 길게 멈추면 `StreamInterrupted`가 납니다 |
| `retry_on_empty` | `True` | 답이 빈 응답을 실패로 보고 다음 모델로 넘깁니다. 스트림이 추론만 보내고 끝난 경우도 포함합니다. `False`면 빈 답을 그대로 돌려줍니다 |
| `allow_training` | `True` | `False`면 학습하지 않는다고 밝힌 모델만 씁니다. OpenRouter에는 `provider.data_collection="deny"`를 보냅니다 |
| `reject_response` | `None` | `fn(ChatResponse) -> bool \| str`. 참이면 그 답을 버리고 다음 모델로 넘어갑니다 (스트림 제외) |
| `store` | `MemoryStore()` | `SQLiteStore("state.db")`로 프로세스 사이에 cooldown과 사용량 공유 |
| `retry_budget` | `True` | `RetryBudget(...)`으로 조정하거나 `False`로 끕니다 |
| `cooldowns` | `CooldownPolicy()` | 종류별 기본 cooldown, 최대 백오프 |
| `on_attempt` | `None` | 시도마다 호출되는 콜백 (`Attempt`) |
| `clock` | `time.time` | 벽시계 초. 상태를 epoch로 저장하므로 단조 시계를 넘기면 안 됩니다 |

`ai.models(require=["tools"])`는 시도 순서, `ai.status()`는 남은 cooldown, `ai.health()`는 모델별 성공률과 지연을 보여줍니다.
기본 전략에서는 `ai.models()`가 묶음 순서까지만 맞고, 같은 묶음 안의 실제 선택은 그때의 쿼터 상태가 정합니다.

## OpenAI SDK 모양으로 쓰기

```python
from freeai.compat import OpenAI

client = OpenAI()   # FreeAI.from_env() 사용, OpenAI(FreeAI(...))도 가능
reply = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
print(reply["choices"][0]["message"]["content"], reply["x_freeai"])
```

응답은 OpenAI 형식의 dict입니다. 실제로 답한 모델은 `model`에 들어 있고, `x_freeai`에는 답한 공급자
(`provider`), SDK가 고른 카탈로그 모델 id(`requested_model`), 시도 횟수(`attempts`)가 들어 있습니다.
`AsyncOpenAI`도 있습니다.

## 로컬 프록시

```bash
freeai serve                              # http://127.0.0.1:8787/v1
freeai --no-keyless serve                 # 익명 엔드포인트 없이 내 키만 (기본은 익명 OVHcloud 포함)
freeai --keyless serve                    # Kilo도 키 없이 사용 (무료 모델이 프롬프트를 학습에 쓸 수 있음)
FREEAI_PROXY_TOKEN=... freeai serve --host 0.0.0.0
```

`/v1/chat/completions`(스트림 포함)와 `/v1/models`, `/health`를 제공합니다. 응답 헤더 `X-FreeAI-Provider`,
`X-FreeAI-Model`이 실제로 답한 모델을 알려 줍니다(스트림은 아래 한계 참고). 모든 모델이 쉬는 중이면 `429`와
`Retry-After`를 돌려줍니다. OpenAI SDK 클라이언트는 `max_retries=0`으로 두세요. 재시도와 전환은 프록시가 이미 합니다.

그 밖의 오류는 OpenAI 형식의 `{"error": {...}}` 본문으로 돌려줍니다. 잘못된 요청, 맞는 무료 모델 없음,
`OutputLimitError`는 `400`, 모든 후보가 실패하면 `503`, 그 밖의 공급자 오류는 `502`입니다. 스트림이 시작된 뒤 난
오류는 `data: {"error": {...}}` 이벤트 하나로 보내고 `data: [DONE]`은 보내지 않습니다. `[DONE]` 없이 끝난 스트림은
실패로 처리하세요.

쿼터를 얼마나 기다릴지는 프록시를 띄운 쪽이 정합니다(`freeai --max-wait 120 serve`). 요청 본문의 `max_wait`는
그보다 짧게만 받아들입니다(`0`이면 기다리지 않고 바로 `429`). 기다리는 요청은 프록시의 스레드 하나를 붙잡고
있으므로, 길게 잡을수록 동시에 기다리는 요청 수만큼 스레드가 쌓입니다.

보안: 프록시는 시작할 때 읽은 모든 API 키의 한도를 대신 씁니다. 그래서 기본 바인딩은 루프백이고, 다른 주소에는
bearer 토큰이 없으면 시작하지 않습니다. 토큰은 상수 시간으로 비교합니다. 오류 본문에서 키 문자열을 가리고(4자
미만인 키는 제외), 요청 본문은 16 MiB로 제한합니다. 클라이언트는 공급자 키를 보내지 않습니다.

## 명령줄

```bash
freeai models --require tools        # 시도 순서대로 후보 출력
freeai models --json                 # 후보 키와 문맥 길이를 JSON으로
freeai audit                         # 고정 모델 목록을 플랫폼 카탈로그와 비교 (--json / --fail-on-catalog-error)
freeai chat "질문" --model smart --stream
freeai chat "질문" --system "짧게 답해" --max-tokens 256
freeai --store ~/.cache/freeai.db chat "..."   # 다른 프로세스와 상태 공유
freeai --tiers free,trial chat "..."            # 체험 크레딧 프리셋까지 허용
freeai serve --port 9000 --token ...            # --token 기본값은 FREEAI_PROXY_TOKEN
freeai --max-wait 120 serve                                   # 쿼터를 최대 2분까지 기다리는 프록시
freeai --strategy priority --no-adapt-limits chat "..."       # 이전 동작: 나열한 순서, 선언 한도 그대로
```

공통 옵션 `--tiers`, `--keyless`, `--no-keyless`, `--no-training`, `--store`, `--strategy`, `--max-wait`, `--no-adapt-limits`, `-v`는
하위 명령(`serve`, `models`, `audit`, `chat`) 앞에 씁니다.

## 관측

```python
from freeai.otel import instrument
instrument(ai)   # 시도마다 GenAI 규약 span 하나 (gen_ai.provider.name, gen_ai.request.model, error.type ...)
```

`opentelemetry-api`가 필요합니다(`free-llm-sdk[otel]`). GenAI 시맨틱 규약이 아직 Development 상태라 실험 기능입니다.
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
- 스트리밍이 아닌 `chat()`은 응답이 끝나야 헤더가 옵니다. 그사이 다른 응답의 헤더가 이 요청을 이미 세고 있으면
  예약분과 겹칩니다. 보낸 지 6초(`freeai.limits.ACCEPT_MARGIN`)가 넘은 요청은 헤더에 들어 있다고 보고 한 번만
  세지만, 그보다 젊은 요청은 아직 가는 중일 수 있어 여전히 한 번 더 셉니다. 6초는 연결 타임아웃 5초에 여유 1초를
  더한 값입니다. Groq처럼 1~2초에 답하는 플랫폼에서는 진행 중인 요청이 대부분 6초보다 젊어서 효과가 거의 없고,
  답이 긴 요청이나 느린 플랫폼에서만 줄어듭니다. 더 줄이려면 요청이 실제로 전송된 시각을 기록해야 하는데 아직
  하지 않았습니다. 연결 타임아웃이 더 긴 클라이언트를 직접 넘기면 6초가 모자랄 수 있습니다.
- `strategy="quota"`의 순서는 선언했거나 헤더로 배운 한도에 기댑니다. 선언이 틀리면 순서도 틀립니다. `uncounted`로
  표시한 Z.ai의 창(하루)과 키를 쓴 Kilo의 창(한 시간)은 확인된 값이 아니라 아껴 쓰기 위한 가정입니다.
- `adapt_limits`는 모의 서버로만 확인했습니다. 대상인 Gemini의 키가 없어 실제 플랫폼에서는 돌려 보지 못했습니다.
  탐침이 거절되면 그 요청은 시도 한 번을 잃고 다음 모델로 넘어갑니다. 같은 창의 탐침을 하나로 묶는 것은 프로세스
  안에서만이라, 프로세스가 여럿이면 동시에 하나씩 보낼 수 있습니다. 플랫폼의 하루가 `reset_tz`와 다른 시각에
  시작하면 집계가 어긋나 한도를 잘못 내릴 수 있고, 그 경우 다음 날 탐침으로 다시 올라갑니다.
- 기본 전략 `"quota"`는 같은 묶음의 후보를 모두 평가하므로 `"priority"`보다 호출당 비용이 큽니다. 후보 50개, 창 4개씩인 모의
  측정에서 호출당 약 0.3 ms가 1.0 ms로(`MemoryStore`), 0.4 ms가 2.3 ms로(`SQLiteStore`) 늘었습니다.
- 이전 버전은 저장소의 창마다 선언 한도를 복사해 두었고, 선언값을 바꿔도 그 복사본을 계속 썼습니다. 이제는
  플랫폼이 알려 준 크기만 저장하고, 예전 복사본은 무시합니다.
- 플랫폼이 요청을 받는 순간 센다고 가정합니다. OVHcloud와 Groq에서 실측으로 확인했습니다
  (`examples/chat_app/probe_headers.py`). 다른 플랫폼은 같은 스크립트로 요청 세 번에 확인할 수 있습니다.
- Groq는 요청을 받는 순간 채팅 템플릿을 포함한 프롬프트와 `max_tokens`를 분당 토큰에서 미리 뺍니다. 템플릿 몫은
  프리셋이 `prompt_overhead_tokens = 80`으로 더합니다(실측 15~74토큰). `max_tokens`가 없으면 Groq가 잡는
  출력 몫이 2~961토큰으로 매번 달라 미리 알 수 없습니다(gpt-oss는 주로 최근 응답 하나의 길이). SDK는 512로
  어림하므로, Groq에서 정확히 맞추려면 `max_tokens`를 주세요. qwen은 분당 출력 토큰 1000 한도가 따로 있어,
  `max_tokens` 없는 요청 12번 중 1번이 `Request too large` 429를 받았습니다. `max_tokens` 1500 요청은 받아들여져서,
  무엇이 이 오류를 부르는지는 모릅니다. Groq가 요청 자체가 크다고 알리므로 SDK는 그 요청에서만 모델을 뺍니다.
- qwen의 분당 출력 토큰 1000은 프리셋이 `otpm(1000)`으로 선언합니다. 요청마다 100토큰을 잡아 두었다가
  `usage.completion_tokens`로 정산하고, OTPM 429 본문의 `Used`로 다시 맞춥니다. Groq가 잡는 출력 몫은 `max_tokens`
  256에 약 50토큰 응답일 때 80~103이었고, 계산식은 모릅니다. 응답이 길면 qwen의 분당 처리량은 그만큼 줄어듭니다.
  이 한도는 qwen에서만 봤습니다. 공식 문서에 없는 값이라 Groq가 바꾸면 프리셋도 따라 바꿔야 합니다.
- Groq 세 모델과 OVHcloud에 분당 120건을 3분 보낸 실측(`max_tokens` 256)에서, 2분째부터 모델별 성공은 gpt-oss가
  분당 24~26건, qwen이 16~28건이고 Groq 429는 3분에 10번이었습니다(`reports/load-test/groq-rpm-refill-real.jsonl`).
  한도를 넘는 몫은 여전히 `ExhaustedError`로 끝납니다(같은 측정에서 356건 중 61건). 남은 429가 왜 생기는지는 확인하지
  못했습니다. 2.1초 간격에서 요청의 도착 시각이 0.1초 넘게 흔들린 경우로 추정합니다.
- Groq 일일 요청은 86.4초보다 촘촘히 보내는 동안 다시 차지 않습니다(실측). 하루 1000건은 사실상 한 번 쓰면
  쉬는 시간에만 돌아오는 몫으로 보세요. 1000건을 실제로 다 쓴 상태는 확인하지 못했습니다. 회복 방식만
  실측했고(`reports/load-test/groq-daily-refill-real.jsonl`), 다 쓴 뒤 86.4초 간격으로 보내는 동작은 단위 테스트로
  확인했습니다.
- OVHcloud의 분당 한도를 기다리면 최대 약 61초가 걸립니다. `max_wait`가 그보다 짧으면 기다리지 않고 다른
  모델로 넘어갑니다.
- Z.ai의 동시 요청 1개 제한, `next_flush_time` 시간대(UTC+8), NVIDIA 무료 엔드포인트 목록은 공식 확인이 안 된 값입니다.
- 프록시 스트림의 `X-FreeAI-Provider`, `X-FreeAI-Model` 헤더는 첫 청크를 보낸 모델입니다. 그 모델이 추론만 보내고
  끝나 다른 모델로 넘어가면 실제로 답한 모델과 다를 수 있습니다. 청크마다 들어 있는 `model`과 `x_freeai.provider`는
  정확합니다.

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
