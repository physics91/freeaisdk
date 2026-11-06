# 샘플 채팅앱과 부하 측정 도구

freeai SDK(`AsyncFreeAI`)로 만든 스트리밍 웹 채팅앱입니다. 같은 앱을 실제 무료 공급자에 붙여 쓰거나, 무료 한도를 흉내 내는 mock 업스트림에 붙여 부하를 측정할 수 있습니다. 측정 결과는 [`reports/샘플 채팅앱 부하 측정.md`](../../reports/샘플%20채팅앱%20부하%20측정.md)에 있습니다.

## 구성

| 파일 | 역할 |
| --- | --- |
| `app.py` | Starlette 앱. `GET /` 웹 UI, `POST /api/chat` SSE 스트리밍, `GET /api/models`, `GET /api/stats` |
| `static/index.html` | 바닐라 JS 채팅 UI. Enter 전송, Shift+Enter 줄바꿈, Escape 중지, 다크 모드 |
| `scenarios.py` | 공급자별 무료 한도 프로필. 실제 SDK 프리셋 클래스를 mock 주소로 연결 |
| `mock_upstream.py` | 무료 한도를 흉내 내는 가짜 업스트림. 응답은 모두 합성 데이터 |
| `loadgen.py` | 가상 사용자 부하 생성기. 닫힌 루프(`--users`)와 열린 루프(`--rate`) |
| `bench.py` | mock과 앱을 띄우고 단계별 부하를 걸어 JSONL로 기록 |
| `probe_headers.py` | 실제 플랫폼이 요청을 받는 순간 세는지, 끝날 때 세는지 한도 헤더로 확인 (요청 3번) |

웹 의존성(Starlette, uvicorn)은 예제 전용입니다. SDK의 `pyproject.toml`과 `uv.lock`에는 들어가지 않고 `uv run --with`로만 받습니다.

```bash
W="uv run --with starlette==1.10.0 --with uvicorn[standard]==0.54.0"
```

## 실제 공급자로 실행

환경 변수의 API 키를 모두 쓰고, `--keyless`를 주면 키 없이 쓰는 OVHcloud와 Kilo도 추가합니다.

```bash
$W python examples/chat_app/app.py --keyless          # http://127.0.0.1:8000
$W python examples/chat_app/app.py --keyless --no-training   # 프롬프트로 학습하는 모델(Kilo 등) 제외
```

- 모든 무료 모델이 한도에 걸리면 `max_wait`(기본 2초)만 기다린 뒤 `429`와 `Retry-After`를 돌려줍니다. UI는 몇 초 뒤 다시 시도하라고 안내합니다.
- `--deadline 45`처럼 요청당 상한을 거는 것을 권합니다. 실측에서 OVHcloud 모델이 첫 토큰을 20초 넘게 보내지 않는 일이 있었고, 상한이 없으면 failover가 이어져 응답이 1분을 넘기기도 했습니다.
- `--pool N`은 mock에 붙일 때만 쓰는 옵션입니다. 공급자들이 함께 쓰는 httpx 연결 풀 크기를 정합니다. 주지 않으면 SDK 기본값(공급자당 연결 1000개, keep-alive 100개)을 씁니다.
- 여러 프로세스로 띄우려면 한도를 공유하도록 `--workers 4 --store state.db`처럼 SQLite 저장소를 지정해야 합니다. 동시 실행 슬롯(`max_concurrency`, Z.ai 기본 1)은 프로세스마다 따로 셉니다.

## 부하 측정

```bash
cd examples/chat_app
$W python bench.py overhead --scenario instant --levels 1,10,50,100,200 --duration 20   # 앱과 SDK 자체 처리량
$W python bench.py overhead --scenario slow --levels 50,100,200,400 --duration 30       # 동시 스트림 수
$W python bench.py capacity --rates 30,60,120,240,480,960 --duration 120                # 무료 한도 시뮬레이션
$W python bench.py soak --rate 60 --duration 600                                        # 일일 한도 소진 곡선
$W python bench.py real --rates 6,12,24 --duration 120                                  # 실제 keyless 공급자
$W python bench.py overhead --levels 200 --pool 0,1000 --repeat 5 --duration 20         # 두 설정을 번갈아 5번씩
```

`overhead`의 `--repeat`는 단계마다 앱과 mock을 새로 띄워 같은 측정을 반복합니다. `--pool`에 값을 여러 개 주면 반복마다 번갈아 재므로, 머신 상태가 서서히 변해도 두 설정이 똑같이 영향을 받습니다. mock은 429를 줄 때 한도까지 몇 ms 모자랐는지 기록하고, 결과의 `upstream_429_shortfall_ms`에 남깁니다.

실제 플랫폼의 차감 시점은 `probe_headers.py`로 확인합니다. 스트림 A를 연 채로 B를 보내 B의 `remaining`이 A를 이미 뺐는지 봅니다. 출력에는 이름에 `ratelimit`이 들어간 헤더와 `retry-after`만 나오고, 키는 나오지 않습니다.

```bash
uv run python examples/chat_app/probe_headers.py ovh --model gpt-oss-20b
GROQ_API_KEY=... uv run python examples/chat_app/probe_headers.py groq --model openai/gpt-oss-20b
```

mock 시나리오:

- `instant`: 지연, 한도, 오류가 없는 공급자 하나. 앱과 SDK의 CPU 비용만 봅니다.
- `slow`: 한도 없이 실제와 비슷한 지연(첫 토큰 0.5~1.5초, 초당 50~100토큰)만 줍니다.
- `all`: OpenRouter, Groq, Gemini, Cloudflare, Z.ai, SambaNova, NVIDIA, Requesty, Kilo, OVHcloud의 공개 무료 한도. 키를 모두 가진 경우를 가정합니다.
- `keyless`: OVHcloud와 Kilo 익명 한도만.

`--window bucket`(기본)은 분 단위 한도를 토큰 버킷으로, `--window sliding`은 더 엄격한 슬라이딩 윈도로 셉니다. Groq와 OVHcloud는 이 옵션과 관계없이 실측한 방식을 따릅니다.

- Groq: 일일 요청과 분당 토큰이 계속 다시 차는 버킷입니다. 실제 Groq 일일 요청은 직전 요청 뒤로 86.4초가 지날 때마다 하나씩 돌아오고 자투리를 버리지만, mock은 연속으로 채웁니다. 2분 측정에서는 차이가 나지 않습니다. 요청을 받을 때 프롬프트(채팅 템플릿 15~74토큰 포함)와 `max_tokens`를 미리 빼고, 응답이 끝나면 쓰지 않은 몫을 돌려줍니다. `max_tokens`가 없으면 gpt-oss는 그 모델의 최근 응답 20개 중 하나의 길이를, qwen은 실제 Groq에서 관측한 출력 몫(2~961토큰) 중 하나를 미리 뺍니다. 실제 gpt-oss의 출력 몫이 최근 응답 길이와 자주 같았기 때문입니다(24번 중 16번). gpt-oss도 최근 기록이 없으면 관측값(16~831토큰)에서 고릅니다.
- OVHcloud: 벽시계 분마다 리셋되는 고정 창입니다. Kong 형식 헤더(`x-ratelimit-remaining-minute`, 정수 초로 올림한 `ratelimit-reset`)와 429 본문 `{"message":"API rate limit exceeded"}`를 보냅니다.

그래서 Groq와 OVHcloud가 들어간 수치는 이 방식을 넣기 전 mock으로 잰 결과와 직접 비교할 수 없습니다. `bench.py`의 `--max-tokens`(기본 256)는 앱이 요청마다 보내는 `max_tokens`이고, 0이면 보내지 않습니다.

## 보안

- 앱은 실행할 때 받은 모든 키의 쿼터를 씁니다. 그래서 루프백 주소에만 바인딩하고 다른 주소는 거부합니다.
- 대화 기록은 프로세스 메모리에만 두고(세션 5000개, 세션당 10턴 LRU) 로그에 남기지 않습니다.
- 모델 출력은 `textContent`로만 화면에 넣습니다. 오류 메시지의 공급자 키 문자열은 가립니다.
- 요청 본문은 64 KiB, 메시지는 8000자, `session_id`는 32자리 16진수로 제한합니다.
