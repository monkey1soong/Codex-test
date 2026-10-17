# Qwen3 local inference on Colab

Hugging Face의 `Qwen/Qwen3-0.6B`를 다운로드하여 로컬 Python 프로세스에서 실행합니다.
Qwen3의 소형 모델은 0.6B이며, 0.5B 모델명은 Qwen2.5 계열에 해당합니다.
Transformer를 처음부터 학습하는 구현이 아니라 사전 학습된 Transformer의 추론 예제입니다.

## Colab 실행

1. Colab의 **런타임 → 런타임 유형 변경 → T4 GPU**를 선택합니다.
2. `run_qwen3.py`와 `requirements.txt`를 Colab 작업 폴더에 업로드합니다.
3. 아래 셀을 실행합니다.

```python
%pip install -r requirements.txt
```

```python
!python run_qwen3.py --prompt "Self-attention이 무엇인지 한국어로 쉽게 설명해줘."
```

대화형 실행은 Colab에서 다음 셀을 사용합니다.

```python
%run run_qwen3.py
```

`/clear`는 대화 기록을 지우고 `/exit`는 종료합니다. 대화가 길어지면 메모리와
모델 컨텍스트 한도에 도달할 수 있으므로 `/clear`로 새 대화를 시작하세요.
GPU가 없으면 CPU로 실행되지만 응답 속도가 느립니다.

## 옵션과 처리 흐름

```bash
python run_qwen3.py --prompt "1+1은?" --max-new-tokens 128
python run_qwen3.py --prompt "단계별로 문제를 풀어줘: 12*13" --thinking
python run_qwen3.py --prompt "안녕" --local-files-only
```

입력 메시지 → 채팅 템플릿 → 토큰화 → Transformer의 다음 토큰 예측 반복 → 텍스트 복원
순서로 동작합니다. 기본값은 thinking 비활성화이며, `--thinking`은 추론 과정도
생성합니다. 토큰 한도에 도달하면 답변이나 추론 과정이 중간에 끝날 수 있습니다.

최초 실행에는 모델 다운로드를 위한 인터넷과 수 GB의 디스크 공간이 필요합니다.
이후 같은 Hugging Face 캐시가 유지되는 환경에서는 `--local-files-only`로 오프라인
실행할 수 있습니다. Colab 런타임이 초기화되면 캐시도 사라질 수 있습니다.
입력 문장은 추론 API로 전송하지 않으며, 모델 다운로드 후 계산은 현재 런타임에서 수행됩니다.
