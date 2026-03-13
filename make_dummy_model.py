import os
import tensorflow as tf

# ==========================================================
# 1. TensorFlow Dummy 모델 구조 정의
# ==========================================================
# 모델 입력 형태 정의
# # 입력: 224x224 RGB 이미지 (채널 3개)
# 출력: 5개의 클래스 확률

inputs = tf.keras.Input(shape=(224, 224, 3))

# GlobalAveragePooling2D: 이미지의 공간 정보를 가로x세로 평균값으로 압축
x = tf.keras.layers.GlobalAveragePooling2D()(inputs)

# Dense(16): 뉴런 16개짜리 완전연결층. 입력값을 받아 학습된 가중치로 계산
# activation="relu": 음수는 0으로, 양수는 그대로 통과시키는 활성화 함수
x = tf.keras.layers.Dense(16, activation="relu")(x)

# Dense(5): 최종 출력층. 5개의 클래스(분류 항목)로 결과를 냄
# activation="softmax": 5개 출력값의 합이 1이 되도록 확률로 변환 (어느 클래스인지 확률 계산)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

# 위에서 정의한 입력과 출력을 연결해 하나의 모델 생성
model = tf.keras.Model(inputs, outputs)


# ==========================================================
# 2. 모델 저장 경로 설정
# ==========================================================

# TensorFlow SavedModel 형식 저장 위치
saved_model_dir = "models/saved_model"

# 최종 TFLite 모델 파일 경로
tflite_path = "models/model.tflite"

# models 디렉토리가 없으면 생성, 이미 있으면 에러 없이 넘어감
os.makedirs("models", exist_ok=True)


# ==========================================================
# 3. TensorFlow SavedModel 저장
# ==========================================================
# Keras 모델을 TensorFlow 표준 모델 포맷(SavedModel)으로 저장

model.export(saved_model_dir)
print("SavedModel created")


# ==========================================================
# 4. SavedModel → TFLite 변환
# ==========================================================
# SavedModel을 TFLite 변환기로 로드

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 실제 변환 수행. 모델을 TFLite 바이너리 형태로 변환
tflite_model = converter.convert()


# ==========================================================
# 5. TFLite 모델 파일 저장
# ==========================================================
# 변환된 모델을 바이너리 파일로 저장

# with 블록이 끝나면 파일이 자동으로 닫힘
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model created at {tflite_path}")
