# SNS 감성 사진 구도 피드백 AI 모델

Android CameraX + TensorFlow Lite 기반 실시간 AI 카메라 앱의 구도 피드백 모델 학습 레포입니다.

카메라 프레임을 실시간으로 분석해 사용자가 SNS 감성 사진 구도를 맞출 수 있도록 단계적 피드백을 제공합니다.

---

## 현재 레포 상태

| 파일/폴더 | 설명 |
|---|---|
| `make_dummy_model.py` | Android 연동 테스트용 더미 모델 생성 스크립트. 실제 학습과 무관 |
| `models/model.tflite` | 위 스크립트로 생성된 더미 TFLite 파일. Android 연동 테스트용으로 보존 |
| `src/inference/` | 추론 서버 코드 (추후 구현 예정) |
| `src/training/` | 실제 학습 파이프라인 코드 (추후 단계적으로 추가 예정) |
| `dataset/` | 학습 데이터 폴더 구조만 관리. 실제 이미지는 Git에 포함하지 않음 |
| `outputs/` | 학습 산출물 폴더 구조만 관리. 실제 파일은 Git에 포함하지 않음 |
| `labels.txt` | 분류 클래스 레이블 (알파벳순) |

> 현재 단계에서는 실제 학습 코드가 없습니다. 폴더 구조와 기반 설정만 완료된 상태입니다.

---

## 분류 클래스

피드백 우선순위 1단계: 상/하/좌/우 구도 피드백

| 클래스 | 의미 |
|---|---|
| `up` | 카메라를 위로 올려야 하는 상태 |
| `down` | 카메라를 아래로 내려야 하는 상태 |
| `left` | 카메라를 왼쪽으로 이동해야 하는 상태 |
| `right` | 카메라를 오른쪽으로 이동해야 하는 상태 |
| `good` | 현재 구도가 적절한 상태 |

---

## 데이터셋 폴더 구조

```
dataset/
├── train/
│   ├── down/
│   ├── good/
│   ├── left/
│   ├── right/
│   └── up/
├── val/
│   ├── down/
│   ├── good/
│   ├── left/
│   ├── right/
│   └── up/
└── test/
    ├── down/
    ├── good/
    ├── left/
    ├── right/
    └── up/
```

실제 이미지 파일은 `.gitignore`로 제외됩니다. 폴더 구조는 `.gitkeep`으로 유지됩니다.

---

## labels.txt 순서 주의

`labels.txt`의 클래스 순서는 `tf.keras.utils.image_dataset_from_directory`가 폴더명을 **알파벳순**으로 읽는 순서와 일치합니다.

```
down
good
left
right
up
```

TFLite 모델의 출력 인덱스와 이 순서가 반드시 매핑되어야 합니다.

---

## 산출물 구조

```
outputs/
├── models/          # 학습된 Keras 모델 (.keras)
├── tflite/          # 변환된 TFLite 모델
├── reports/         # 평가 결과 (confusion matrix, accuracy 등)
└── android_assets/  # Android 앱에 넣을 최종 파일 (model.tflite, labels.txt)
```

실제 산출물 파일은 `.gitignore`로 제외됩니다. 폴더 구조는 `.gitkeep`으로 유지됩니다.

---

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```
