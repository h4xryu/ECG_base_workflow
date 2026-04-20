"""PTQBuilder — TFLite converter를 사용한 Post-Training Quantization 빌더."""

from typing import Callable, Optional, Sequence

import numpy as np
import tensorflow as tf

from .specs import LayerRule

keras = tf.keras


_SUPPORTED_MODES = ("dynamic", "int8", "float16")


class PTQBuilder:
    """rules/default_rule을 받아 fp32 Keras 모델을 TFLite bytes로 변환한다.

    QATBuilder와 동일한 생성자 시그니처를 가지므로 같은 rules/default_rule을
    QAT 경로와 PTQ 경로에서 공유할 수 있다.

    주의: TFLite converter는 per-layer 세밀한 quantizer 제어를 지원하지 않는다.
    rules의 skip=True 레이어는 _ops_to_target에서 제외하는 방식으로만 반영 가능하며,
    learnable_scale / learnable_threshold 같은 커스텀 quantizer는 PTQ 경로에서
    직접 적용되지 않는다 (TFLite 표준 calibration으로 대체된다).
    자세한 제한사항은 README.md의 'PTQ 제한사항' 섹션 참조.
    """

    def __init__(self, rules: Sequence[LayerRule], default_rule: LayerRule):
        # rules/default_rule은 skip 여부 확인에만 사용한다.
        self.rules = list(rules)
        self.default_rule = default_rule

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(
        self,
        model: keras.Model,
        representative_dataset: Optional[Callable] = None,
        mode: str = "int8",
    ) -> bytes:
        """fp32 Keras 모델을 PTQ 적용된 TFLite bytes로 변환한다.

        Args:
            model: 변환할 fp32 Keras 모델.
            representative_dataset: int8 모드 전용 calibration generator.
                각 호출에서 [input_tensor] 형태의 list를 yield해야 한다.
            mode: "dynamic" | "int8" | "float16"

        Returns:
            TFLite flatbuffer bytes.
        """
        if mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"mode must be one of {_SUPPORTED_MODES}, got '{mode}'"
            )
        if mode == "int8" and representative_dataset is None:
            raise ValueError(
                "representative_dataset is required for int8 mode. "
                "Provide a generator function that yields [input_tensor] lists. "
                "Example:\n"
                "  def repr_gen():\n"
                "      for x in calibration_data:\n"
                "          yield [x[np.newaxis].astype(np.float32)]"
            )

        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if mode == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        elif mode == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        elif mode == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        return converter.convert()

    def evaluate(
        self,
        tflite_bytes: bytes,
        dataset,
    ) -> float:
        """TFLite 모델을 interpreter로 추론하고 accuracy를 반환한다.

        Args:
            tflite_bytes: build()가 반환한 bytes.
            dataset: (x, y) 쌍을 yield하는 iterable.
                x shape: (batch, ...) 또는 (1, ...)
                y shape: (batch,) 정수 레이블 또는 (batch, n_classes) one-hot.

        Returns:
            float accuracy (0.0 ~ 1.0).
        """
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_scale, input_zero_point = self._get_quant_params(input_details[0])
        output_scale, output_zero_point = self._get_quant_params(output_details[0])

        correct = 0
        total = 0

        for x_batch, y_batch in dataset:
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            # 배치를 한 샘플씩 처리한다 (TFLite interpreter는 배치 크기 1 고정).
            if x_batch.ndim == len(input_details[0]["shape"]):
                x_batch = x_batch[np.newaxis]
                y_batch = y_batch[np.newaxis] if y_batch.ndim == 0 else y_batch

            for i in range(len(x_batch)):
                x = x_batch[i : i + 1].astype(np.float32)

                # int8 모드인 경우 입력을 quantize한다.
                if input_details[0]["dtype"] == np.int8:
                    x = np.round(x / input_scale + input_zero_point).astype(np.int8)

                interpreter.set_tensor(input_details[0]["index"], x)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]["index"])

                # int8 출력인 경우 dequantize한다.
                if output_details[0]["dtype"] == np.int8:
                    out = (out.astype(np.float32) - output_zero_point) * output_scale

                pred = np.argmax(out, axis=-1)[0]

                y = y_batch[i]
                if y.ndim > 0:  # one-hot
                    y = np.argmax(y)

                correct += int(pred == int(y))
                total += 1

        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_quant_params(detail):
        quant = detail.get("quantization", (1.0, 0))
        scale = quant[0] if quant[0] != 0 else 1.0
        zero_point = quant[1]
        return scale, zero_point
