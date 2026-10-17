# Model_test: SmolLM2-360M-Instruct

Hugging Face의 [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)를 로컬에서 실행하고 지시 이행과 토큰 생성 동작을 살펴보는 예제입니다. 모든 추론 코드는 `SmolLM2-360M-Instruct.py` 한 파일에 있습니다.

## Colab 실행

Colab에서 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다. Python 파일과 `requirements.txt`를 작업 폴더에 업로드한 다음 실행하세요.

```python
%pip install -r requirements.txt
!python SmolLM2-360M-Instruct.py
```

Python 파일만 업로드했다면 직접 패키지를 설치해도 됩니다.

```python
%pip install "torch>=2.2" "transformers>=4.46,<5" "safetensors>=0.4"
!python SmolLM2-360M-Instruct.py --compare --show-tokens
```

GPU가 없으면 CPU로 자동 실행합니다. 최초 실행에는 인터넷과 모델 다운로드 공간이 필요하며 이후에는 Hugging Face 캐시를 재사용합니다. 입력 문장은 추론 API에 보내지 않고 현재 런타임에서 처리합니다. 모델이 캐시에 모두 있으면 `--local-files-only`로 실행할 수 있습니다.

## 테스트 항목

| 이름 | 지시 내용 | 평가 방식 |
| --- | --- | --- |
| `instruction` | 감정 분류를 소문자 단어 하나로 응답 | `positive`와 정확히 일치 |
| `arithmetic` | 12 × 7을 정수 하나로 응답 | `84`와 정확히 일치 |
| `json` | Alice의 이름과 나이를 JSON으로 추출 | JSON 파싱, 정확한 키·값·타입 확인 |
| `summary` | 사실 추가 없이 25단어 이하 한 문장 요약 | 사람이 내용과 형식 확인 |
| `code` | 대소문자를 무시하는 모음 개수 함수 작성 | 사람이 확인; 생성 코드는 자동 실행하지 않음 |
| `korean` | 토큰의 의미를 한국어 두 문장으로 설명 | 사람이 언어·내용·형식 확인 |

SmolLM2는 주로 영어를 대상으로 하는 소형 모델이므로 영어 테스트를 기본으로 하고 한국어 항목은 추가 관찰용으로 포함했습니다. `fail`은 해당 출력이 지시된 정답/형식을 충족하지 않았다는 의미이며 프로그램 오류가 아닙니다. 자유 응답은 `manual_review`로 표시합니다. 이 작은 예제 집합의 통과 여부를 종합적인 모델 성능으로 해석하지 마세요.

```python
# 같은 지시문을 greedy와 sampling으로 비교
!python SmolLM2-360M-Instruct.py --compare

# JSON 지시만 실행
!python SmolLM2-360M-Instruct.py --case json --show-tokens

# 사용자 지정 instruction
!python SmolLM2-360M-Instruct.py --prompt "Explain tokens in three short bullet points." --max-new-tokens 200

# 샘플링의 무작위성과 seed를 바꾸어 관찰
!python SmolLM2-360M-Instruct.py --case summary --mode sample --temperature 0.8 --top-p 0.95 --seed 7
```

## 측정 결과 읽기

콘솔 표와 `results/smollm2_results.json`에 다음 내용을 기록합니다.

- 실제 응답, 입력 토큰 수, **프롬프트를 제외한 생성 토큰 수**(EOS 등 특수 토큰 포함)
- 생성 시간과 초당 생성 토큰 수, EOS 종료 여부, 출력 토큰 상한 도달 여부
- 출력 토큰 ID와 토큰 조각, 고유 토큰 비율과 반복된 3-token 구간 비율
- 자동 채점 또는 수동 검토 상태, 장치·dtype·라이브러리 버전·모델 커밋 정보
- GPU 사용 시 최대 PyTorch 할당 메모리(MiB, 모델 가중치 포함; 전체 GPU 메모리와 다름)

토큰 조각은 읽기 쉬운 문자열이나 완전한 문자와 일치하지 않을 수 있습니다. 고유 토큰 비율은 높을수록 정답이라는 점수가 아니고, 반복 비율도 반복 생성 현상을 살펴보는 참고치입니다.

시간은 1토큰 warmup 후 `model.generate` 전체를 측정하며 입력 처리(prefill)를 포함합니다. 모델 로딩·다운로드, 토큰화, 텍스트 복원 시간은 제외하고 CUDA는 전후 동기화합니다. 첫 토큰 지연(TTFT)이나 순수 디코딩 속도 측정은 아닙니다. 응답 길이가 다르면 tokens/s도 달라지므로 장치와 옵션을 고정하고 반복 비교하세요. Seed는 고정하지만 장치와 라이브러리가 다르면 결과가 달라질 수 있습니다.

`hit_token_limit=true`이면 응답이 중간에 끊겼을 수 있으므로 `--max-new-tokens`를 늘려 다시 확인하세요. 입력과 최대 출력의 합이 모델 컨텍스트 한도를 넘으면 자동으로 자르지 않고 오류를 표시합니다.

결과 파일은 실행할 때 같은 경로를 덮어쓰며 각 항목 성공 후 갱신됩니다. 비교 결과를 따로 보관하려면 `--output results/experiment_01.json`처럼 경로를 지정하세요. 프롬프트와 응답도 저장되므로 자신의 실험 데이터를 보관할 위치를 선택하세요. Colab 런타임 초기화 전 결과 파일을 다운로드하거나 Google Drive 경로로 저장할 수 있습니다.

```python
from google.colab import files
files.download("results/smollm2_results.json")
```

`--revision`에 Hugging Face 모델 커밋 SHA를 지정하면 모델 버전까지 고정할 수 있습니다. 전체 옵션은 `python SmolLM2-360M-Instruct.py --help`를 참고하세요.
