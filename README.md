# ECG_base_workflow

## 주요 기능

- **데이터 로딩**: MIT-BIH Arrhythmia Database에서 ECG 데이터를 로드하고 전처리
- **모델 학습**: 커스텀 트레이너 클래스를 통한 효율적인 모델 학습
- **평가 및 시각화**: 모델 성능 평가 및 t-SNE를 통한 특징 시각화
- **결과 저장**: 학습 결과 및 메트릭을 Excel 파일로 저장

### 기본 실행

```bash
python main.py
```

1. 데이터 로드 및 전처리
2. 모델 학습
3. 평가 및 결과 저장

### 커스텀 설정

`config.py`

- 모델 아키텍처
- 학습 하이퍼파라미터
- 데이터 경로
- 결과 저장 경로

## 프로젝트 구조

```
ECG_base_workflow/
├── main.py              # 메인 실행 파일
├── config.py            # 설정 파일
├── dataloader.py        # 데이터 로딩 모듈
├── batchloader.py       # 배치 데이터 생성
├── model.py             # 모델 아키텍처 정의
├── modules.py           # 커스텀 레이어 및 모듈
├── train.py             # 학습 함수
├── trainer.py           # 커스텀 트레이너 클래스
├── eval.py              # 평가 및 시각화
├── metrics.py           # 메트릭 계산
├── loss.py              # 손실 함수
├── logger.py            # 로깅 유틸리티
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
