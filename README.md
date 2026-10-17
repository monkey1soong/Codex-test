# Transformer 실습

- `qwen3_test/`: 사전 학습된 Qwen3-0.6B 추론 예제. 폴더 안의 README를 참고하세요.
- `train.py`: 작은 decoder-only Transformer를 무작위 가중치부터 학습하는 단일 파일입니다. Qwen3 미세 조정이 아닌 문자 단위 다음 토큰 예측 실습입니다.

## Colab 실행

GPU 런타임을 선택하고 `train.py`를 업로드한 뒤 실행합니다.

```python
%pip install "torch>=2.2"
!python train.py --steps 500
```

별도 데이터가 없으면 내장 한국어 예제를 사용합니다. 문자/위치 임베딩 → causal multi-head attention → feed-forward → 다음 문자 예측을 학습합니다. 학습/검증 손실과 검증 perplexity를 출력하고 `transformer.pt`를 저장한 뒤 텍스트를 생성합니다.

내장 데이터는 학습 과정을 확인하기 위한 작은 반복 데이터입니다. 검증 데이터에도 반복 문장이 있어 검증 수치는 일반화 성능을 의미하지 않습니다. 일반적인 질문에 답하는 모델을 만들려면 훨씬 많은 다양한 데이터와 학습이 필요합니다.

내 UTF-8 텍스트를 업로드해 학습할 수도 있습니다.

```python
!python train.py --data corpus.txt --steps 2000 --prompt "안녕" --output /content/transformer.pt
```

데이터는 앞 90%/뒤 10%로 분리합니다. 각 구간은 `block-size + 1`개 이상의 문자가 필요합니다. 기본 설정에서는 총 약 650자 이상이 필요합니다. 프롬프트에는 학습 데이터에 있는 문자만 사용할 수 있습니다.

저장한 모델로 생성만 실행하려면:

```python
!python train.py --load /content/transformer.pt --prompt "안녕" --generate-tokens 200
```

`--load`는 생성 전용이며 학습 재개가 아닙니다. 직접 만든 신뢰할 수 있는 체크포인트만 사용하세요. Colab 초기화 전에 체크포인트를 다운로드하거나 Drive에 저장하세요. CPU도 지원하지만 느립니다.

빠른 실행 확인:

```python
!python train.py --steps 2 --eval-batches 1 --batch-size 2 --block-size 16 --width 32 --heads 2 --layers 1 --generate-tokens 20
```

`--temperature`는 생성 무작위성, `--block-size`는 한 번에 보는 문자 수, `--width`, `--heads`, `--layers`는 모델 크기를 조절합니다. `python train.py --help`로 옵션을 확인하세요.
