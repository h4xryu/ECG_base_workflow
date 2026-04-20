# easyquant

TensorFlow/Keras 프레임워크에 QAT(Quantization-Aware Training)와
PTQ(Post-Training Quantization)를 최소 침습적으로 끼워 넣는 plug-in 패키지.

---

## 1. 설치

별도 pip 설치 불필요. `easyquant/` 디렉토리를 프로젝트 루트에 복사하면 끝.

```
your_project/
├── easyquant/   ← 이 디렉토리 통째로 복사
├── train.py
└── ...
```

**의존성** (별도 설치 필요):

```
pip install tensorflow tensorflow-model-optimization numpy
```

---

## 2. 최소 예제 — QAT

```python
from easyquant import LayerRule, QuantizerSpec, QATBuilder, LRSchedulerCallback, SnapshotSaver

rules = [
    LayerRule(
        target_types=[tf.keras.layers.Conv1D],
        weight_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
        activation_quantizer=QuantizerSpec("learnable_threshold", {"num_bits": 4}),
    ),
]
default_rule = LayerRule(
    weight_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
    activation_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
)

qat_model = QATBuilder(rules, default_rule).build(fp_model)
qat_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

schedule = CosineRestartSchedule(lr_max=1e-3, lr_min=1e-5, cycle_length=10)
qat_model.fit(
    x_train, y_train,
    epochs=40,
    callbacks=[LRSchedulerCallback(schedule), SnapshotSaver("./snapshots", cycle_length=10)],
)
```

---

## 3. 최소 예제 — PTQ

```python
from easyquant import PTQBuilder

ptq_builder = PTQBuilder(rules, default_rule)

def repr_gen():
    for x in calibration_samples:          # numpy array, shape (n, ...)
        yield [x[np.newaxis].astype(np.float32)]

tflite_bytes = ptq_builder.build(fp_model, representative_dataset=repr_gen, mode="int8")

# 파일로 저장
with open("model_int8.tflite", "wb") as f:
    f.write(tflite_bytes)

# accuracy 평가
acc = ptq_builder.evaluate(tflite_bytes, test_dataset)   # (x, y) iterable
print(f"PTQ int8 accuracy: {acc:.4f}")
```

---

## 4. QAT와 PTQ를 같은 config로 비교 실험

```python
import numpy as np
import tensorflow as tf
from easyquant import (
    LayerRule, QuantizerSpec,
    QATBuilder, PTQBuilder,
    LRSchedulerCallback, SnapshotSaver, CosineRestartSchedule,
)

layers = tf.keras.layers

# ── 공통 config ──────────────────────────────────────────────────
rules = [
    LayerRule(
        target_types=[layers.Conv1D, layers.Conv2D],
        weight_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
        activation_quantizer=QuantizerSpec("learnable_threshold", {"num_bits": 4}),
    ),
    LayerRule(
        target_types=[layers.Dense],
        weight_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
        activation_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
    ),
    LayerRule(name_contains=["softmax", "logits"], skip=True),
]
default_rule = LayerRule(
    weight_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
    activation_quantizer=QuantizerSpec("learnable_scale", {"num_bits": 8}),
)

# ── 경로 1: QAT ──────────────────────────────────────────────────
qat_model = QATBuilder(rules, default_rule).build(fp_model)
qat_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
schedule = CosineRestartSchedule(lr_max=1e-3, lr_min=1e-5, cycle_length=10)
qat_model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=40,
    callbacks=[LRSchedulerCallback(schedule), SnapshotSaver("./snapshots", cycle_length=10)],
)
_, qat_acc = qat_model.evaluate(x_test, y_test)

# ── 경로 2: PTQ (비교군) ─────────────────────────────────────────
def repr_gen():
    for x in x_train[:200]:
        yield [x[np.newaxis].astype(np.float32)]

ptq_builder = PTQBuilder(rules, default_rule)
tflite_bytes = ptq_builder.build(fp_model, representative_dataset=repr_gen, mode="int8")

test_ds = zip(x_test, y_test)    # (x, y) iterable
ptq_acc = ptq_builder.evaluate(tflite_bytes, test_ds)

print(f"QAT  accuracy: {qat_acc:.4f}")
print(f"PTQ  accuracy: {ptq_acc:.4f}")

# ── 경로 3: Quantization 완전 비활성화 ──────────────────────────
# easyquant import 없음. fp_model 그대로 사용.
```

---

## 5. 기존 training loop에 끼워넣는 패턴

기존 `model.fit` 스타일 코드에 QAT를 추가하는 최소 변경 패턴.

```python
# 기존 코드 (변경 없음)
model = build_model()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# ── 이 블록만 추가 ────────────────────────────────────────────────
USE_QAT = True
if USE_QAT:
    from easyquant import LayerRule, QuantizerSpec, QATBuilder, LRSchedulerCallback, SnapshotSaver, CosineRestartSchedule
    rules = [...]
    default_rule = LayerRule(...)
    model = QATBuilder(rules, default_rule).build(model)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    extra_callbacks = [
        LRSchedulerCallback(CosineRestartSchedule(1e-3, 1e-5, 10)),
        SnapshotSaver("./snapshots", cycle_length=10),
    ]
else:
    extra_callbacks = []
# ─────────────────────────────────────────────────────────────────

model.fit(x_train, y_train, epochs=40, callbacks=[...] + extra_callbacks)
```

`USE_QAT = False`로 설정하면 easyquant import 자체가 실행되지 않는다.

---

## 6. 분리 방법

1. `easyquant/` 디렉토리를 삭제한다.
2. 사용 측 코드에서 `from easyquant import ...` 라인을 제거한다.
3. `QATBuilder.build(model)` / `PTQBuilder.build(model, ...)` 호출 부분을 원래 `model`로 교체한다.

프레임워크 내부 코드는 수정할 필요가 없다.

---

## 7. PTQ 제한사항

### per-layer quantizer 제어 불가

TFLite converter는 레이어 단위의 커스텀 quantizer(예: `learnable_scale`,
`learnable_threshold`)를 직접 지원하지 않는다. PTQ 경로에서 `rules`에 담긴
quantizer 설정(weight_quantizer, activation_quantizer 등)은 실질적으로 무시되며,
TFLite 표준 calibration 알고리즘이 대신 적용된다.

현재 `rules`에서 PTQ가 반영하는 설정은 **`skip=True` 유무뿐**이다. 단,
TFLite converter 수준에서도 특정 레이어만 quantization에서 제외하는 공식 API가
없으므로, skip 설정 역시 현재는 참고 정보로만 활용된다.

### mode별 제한

| mode      | weight | activation | 비고 |
|-----------|--------|------------|------|
| dynamic   | int8   | float32    | representative_dataset 불필요 |
| int8      | int8   | int8       | representative_dataset 필수 |
| float16   | fp16   | float32    | GPU 추론 시 속도 이점 |

### evaluate() 제한

- `PTQBuilder.evaluate()`는 배치를 내부적으로 sample 단위로 분해해 처리한다
  (TFLite interpreter의 배치 크기 1 고정 제약).
- dataset은 `(x, y)` 쌍을 yield하는 iterable이어야 한다.
- multi-input 모델은 지원하지 않는다.

### int8 입출력 타입

`int8` 모드로 변환된 모델의 입출력 dtype은 `tf.int8`이다. `evaluate()`에서
자동으로 quantize/dequantize를 처리하지만, 직접 interpreter를 사용할 때는
입력을 `int8`로 변환해야 한다.
