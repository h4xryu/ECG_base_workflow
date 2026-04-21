# ECG_base_workflow

## 주요 기능

- **데이터 로딩**: MIT-BIH Arrhythmia Database에서 ECG 데이터를 로드하고 전처리
- **모델 학습**: 커스텀 트레이너 클래스를 통한 효율적인 모델 학습
- **평가 및 시각화**: 모델 성능 평가 및 t-SNE를 통한 특징 시각화
- **결과 저장**: 학습 결과 및 메트릭을 Excel 파일로 저장
- **양자화 실험**: FP32 / QAT / QAT+Snapshot Ensemble / PTQ 비교 (`autoexp.py`)

## 환경

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 (WSL2 on Windows 11) |
| Python | 3.10.12 |
| CUDA | 12.3 |
| cuDNN | 8.9.x (nvidia-cudnn-cu12) |
| TensorFlow | 2.16.1 |
| numpy | <2.0.0 |

## 설치

### 1. WSL2 + CUDA 환경 설정 (Windows)

Windows에 NVIDIA 드라이버(591.86 이상)가 설치되어 있어야 합니다.

WSL2 터미널에서 CUDA Toolkit 설치:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-3
```

환경변수 설정:

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. Python 패키지 설치

```bash
pip install -r requirements.txt

# cuDNN (pip으로 설치 시)
pip install nvidia-cudnn-cu12==8.9.7.29

# cuDNN 경로 등록
CUDNN_PATH=$(python3 -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')")
echo "export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

### 3. GPU 인식 확인

```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

> **주의**: TensorFlow 2.16은 Keras 3.x를 사용합니다. `tensorflow-model-optimization`(tfmot) 호환을 위해
> `autoexp.py`에 `os.environ['TF_USE_LEGACY_KERAS'] = '1'`이 설정되어 있습니다.

## 실행

### 기본 학습

```bash
python main.py
```

### 양자화 비교 실험 (FP32 / QAT / Snapshot Ensemble / PTQ)

```bash
python autoexp.py
```

결과는 `./results/autoexp/` 에 저장됩니다.

## 프로젝트 구조

```
Classification_workflow/
├── main.py              # 기본 학습 실행
├── autoexp.py           # 양자화 비교 실험
├── config.py            # 설정 파일
├── dataloader.py        # 데이터 로딩 모듈
├── batchloader.py       # 배치 데이터 생성
├── model.py             # 모델 아키텍처 정의
├── modules.py           # 커스텀 레이어 (CATNet, ChannelAttention)
├── train.py             # 학습 함수
├── trainer.py           # 커스텀 트레이너 클래스
├── eval.py              # 평가 및 시각화
├── metrics.py           # 메트릭 계산
├── loss.py              # 손실 함수
├── logger.py            # 로깅 유틸리티
├── easyquant/           # QAT/PTQ 플러그인 패키지
├── requirements.txt     # 의존성 목록
├── mit-bih-arrhythmia-database-1.0.0/  # 데이터셋
├── results/             # 학습 결과 저장
└── logs/                # 로그 파일
```

## 평가 메트릭

- 정확도 (Accuracy)
- 정밀도 (Precision)
- 재현율 (Recall)
- F1-Score
- 혼동 행렬 (Confusion Matrix)

## 결과 시각화

- 학습 곡선 (손실 및 정확도)
- 혼동 행렬 히트맵
- t-SNE를 통한 특징 분포 시각화
