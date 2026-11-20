import tensorflow as tf
import numpy as np
from PIL import Image
import logging
import traceback

from keras._tf_keras.keras.layers import TFSMLayer

# 클래스 이름 정의
CLASS_NAMES = [
    '습진',
    '사마귀, 전염성 연체종 및 기타 바이러스 감염',
    '흑색종',
    '아토피 피부염',
    '기저세포암(BCC)',
    '멜라닌세포 모반(NV)',
    '양성 각화증 유사 병변(BKL)',
    '건선 사진 편평태선 및 관련 질환',
    '지루성 각화증 및 기타 양성 종양',
    '백선, 칸디다증 및 기타 진균 감염'
]

model = TFSMLayer("Model/saved_model_folder", call_endpoint="serving_default")

def preprocess_image(image: Image.Image) -> tf.Tensor:

    if image.mode != "RGB":
        try:
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"RGB 변환 실패: {str(e)}")

    image = image.resize((300, 300))
    image = np.array(image).astype(np.float32)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return tf.convert_to_tensor(image, dtype=tf.float32)

# 예측 함수
def predict_image(image: Image.Image) -> dict:

    try:
        input_tensor = preprocess_image(image)
        output = model(input_tensor)

        # 출력이 dict일 경우, 첫 번째 텐서 가져오기
        if isinstance(output, dict):
            output_tensor = list(output.values())[0]
        elif isinstance(output, tf.Tensor):
            output_tensor = output
        else:
            raise ValueError("예측 결과 형식 오류")

        predictions = output_tensor.numpy()[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(np.max(predictions))

        logging.info(f"예측 결과: {predicted_class} ({confidence:.4f})")

        return {
            "class": predicted_class,
            "confidence": round(confidence, 4),
            "index": predicted_index
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {
            "error": f"서버 오류: {str(e)}"
        }
