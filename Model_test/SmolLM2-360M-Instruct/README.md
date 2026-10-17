# Colab에서 SmolLM2-360M-Instruct 테스트

[Colab 노트북 열기](https://colab.research.google.com/github/monkey1soong/Codex-test/blob/main/Model_test/SmolLM2-360M-Instruct/colab.ipynb)

`model.py`는 단독 실행 파일입니다. 필요한 패키지를 자동 설치하고 Hugging Face 로그인,
GitHub 사용자·저장소 접근 확인, 저장소 clone 또는 재사용, 모델 다운로드와 생성 테스트를
순서대로 실행합니다. 별도의 Python 모듈을 업로드할 필요는 없습니다.

## 처음 한 번 준비하기

1. 위 노트북을 열고 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다. CPU도 지원합니다.
2. Colab 왼쪽 **Secrets(열쇠 아이콘)**에 아래 두 항목을 저장하고 **노트북 액세스**를 켭니다.

   | 이름 | 용도와 권한 |
   | --- | --- |
   | `HF_TOKEN` | Hugging Face 로그인용 읽기 토큰. [토큰 설정](https://huggingface.co/settings/tokens) |
   | `GITHUB_TOKEN` | `monkey1soong/Codex-test`에 접근 가능한 GitHub 토큰. Fine-grained 토큰은 대상 저장소와 **Contents: Read-only** 권한을 선택합니다. [토큰 설정](https://github.com/settings/personal-access-tokens) |

3. 노트북의 **인증 준비** 셀을 실행합니다. Secrets를 실행 환경에 전달하고 GitHub에서 최신 `model.py`를 `/content/model.py`로 내려받습니다.
4. 실행 셀에서 다음 명령을 실행합니다. 이후 같은 런타임에서는 이 명령만 다시 실행하면 됩니다.

```python
!python model.py
```

`!python`은 노트북 커널과 별도 프로세스이므로 `google.colab.userdata.get()`으로
Secrets를 직접 읽을 수 없습니다. 인증 준비 셀에서 전달한 환경 변수를 Python 파일이
읽어 실제 로그인을 수행합니다. 런타임을 초기화하면 준비 셀을 다시 실행하세요.
토큰 값은 코드나 명령어에 적지 않습니다.

기존 노트북에서 파일을 직접 업로드한 경우에는 아래 셀을 먼저 실행할 수도 있습니다.

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")
```

그 다음 `model.py`가 있는 디렉터리에서 `!python model.py`를 실행하세요.

## 자동 실행되는 작업

- 설치된 버전을 확인하고 필요한 PyTorch, Transformers, Hugging Face Hub, Safetensors만 설치합니다.
- `huggingface_hub.login()`으로 Hugging Face 토큰을 확인합니다.
- GitHub API로 로그인 사용자와 `monkey1soong/Codex-test` 접근 권한을 확인합니다.
- 기본적으로 `/content/Codex-test`에 저장소를 clone합니다. 이미 해당 저장소 안에서 실행하거나 대상 경로에 올바른 clone이 있으면 재사용하며, 로컬 변경을 보존하기 위해 자동 pull/reset은 하지 않습니다.
- `HuggingFaceTB/SmolLM2-360M-Instruct`를 다운로드하여 GPU 또는 CPU에서 추론합니다.
- 결과를 현재 작업 폴더의 `results/smollm2_results.json`에 저장합니다.

GitHub 토큰은 Git 인증 도우미에 환경 변수로 전달하며 remote URL이나 Git 설정 파일에
삽입하지 않습니다. Hugging Face 로그인은 라이브러리의 표준 런타임 캐시를 사용합니다.
코드는 GitHub에 결과를 자동 push하지 않으며 저장소 읽기 권한이면 충분합니다.

## 테스트와 옵션

기본 테스트는 영어 지시 이행, 산술, JSON 추출, 요약, Python 코드 생성, 한국어 설명
6개입니다. 앞의 세 항목은 정답/형식을 자동 채점하고 나머지는 `manual_review`로
표시합니다. 생성한 Python 코드는 자동 실행하지 않습니다.

```python
# 같은 지시문을 greedy와 sampling으로 비교
!python model.py --compare --show-tokens

# 내 instruction만 테스트
!python model.py --prompt "Explain tokens in three short bullet points." --max-new-tokens 200

# JSON 생성 테스트만 실행
!python model.py --case json

# 설치 단계 생략: 필요한 버전이 이미 설치된 경우
!python model.py --no-install
```

다른 저장소나 디렉터리를 사용하려면 `--github-repo owner/repo`, `--github-branch main`,
`--repo-dir /content/my-checkout`을 지정할 수 있습니다. 모든 옵션은
`!python model.py --help`로 확인합니다.

JSON에는 응답, 입력/출력 토큰 수, 출력 토큰 ID와 조각, 생성 시간, 초당 생성 토큰 수,
EOS 종료 여부, 토큰 상한 도달 여부, 반복 토큰 지표와 GPU 메모리 사용량을 기록합니다.
인증 토큰은 기록하지 않습니다. 출력 토큰 수에는 EOS 등 특수 토큰이 포함됩니다.
생성 시간은 warmup 후 `model.generate()`를 측정하며 prefill을 포함하고 다운로드·토큰화는
제외합니다. 첫 토큰 지연이나 순수 디코딩 속도 측정은 아닙니다.

SmolLM2는 주로 영어를 대상으로 하는 소형 모델이며 이 테스트는 정식 성능 벤치마크가
아닙니다. `hit_token_limit=true`이면 응답이 잘렸을 수 있으므로 토큰 상한을 늘려보세요.
결과 파일은 같은 경로에 덮어쓰므로 실험을 보관할 때는 `--output results/run2.json`처럼
경로를 바꾸거나 런타임 초기화 전에 다운로드하세요.

```python
from google.colab import files
files.download("/content/results/smollm2_results.json")
```

## 오류 확인

- `Missing HF_TOKEN/GITHUB_TOKEN`: Secrets 액세스를 켜고 인증 준비 셀을 다시 실행합니다.
- `Hugging Face login failed`: 토큰 읽기 권한과 Hugging Face 연결을 확인합니다.
- `GitHub API HTTP 401/403/404`: 토큰 만료, 대상 저장소 선택·읽기 권한, 네트워크 접근을 확인합니다.
- 기존 디렉터리 충돌: 다른 `--repo-dir`를 지정합니다. 기존 파일은 덮어쓰지 않습니다.

클라우드 개발 환경의 네트워크 정책과 Colab 런타임의 네트워크 정책은 별개입니다.
