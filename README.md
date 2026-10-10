# SEAnergy

머신러닝 기반 전기차 배터리 SOH(State of Health) 진단 플랫폼

계명대학교 미래자동차공학전공 종합설계 프로젝트입니다. 전기차 충전 데이터로 배터리 SOH를 예측하고, 그 결과를 웹 대시보드로 보여주는 것을 목표로 합니다.

## 배경

전기차가 늘면서 중고 전기차 거래에서 배터리 상태를 확인하려는 수요도 커졌습니다. 그런데 시중의 OBD2 진단기는 대부분 고장 코드를 읽는 수준이라 배터리가 실제로 얼마나 닳았는지는 알기 어렵습니다. 이 프로젝트는 공개된 배터리 열화 데이터를 학습한 모델로 이 부분을 채우려는 시도입니다.

## 진행 상황

| 단계 | 상태 |
|---|---|
| 데이터 수집·전처리 (NASA, BatICM, SNL) | 완료 |
| SOH 예측 모델 (XGBoost) | 완료 → [아래](#soh-예측-모델) |
| 모델 API · 웹 대시보드 | 진행 예정 |
| 실차 OBD2 연동 | 겨울 Bellevue College 공동 프로젝트에서 진행 예정 |

## 데이터셋

| 데이터셋 | 화학종 | 내용 |
|---|---|---|
| [NASA PCoE](https://www.kaggle.com/datasets/ckskaggle/li-ion-battery-dataset-from-nasa-pcoe) | LCO | 실험실 가속열화, 셀 34개 |
| [BatICM](https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles) | NCM | 실차 20대(BAIC EU500), 29개월 충전 로그 (메인) |
| [SNL](http://batteryarchive.org) | NMC | 온도·방전심도·C-rate 통제 실험, 시험조건 8가지 |

원본 데이터는 용량 때문에 저장소에 넣지 않았고, 세 데이터를 한 표로 합친 학습용 통합 데이터(`모든 데이터 병합.xlsx`, 약 6MB, 71,080행)만 저장소 루트에 포함했습니다. BatICM을 쓰실 때는 원 논문(Applied Energy, 2023)을 인용해 주세요.

## SOH 예측 모델

NASA, BatICM, SNL **세 데이터를 합쳐서** XGBoost 하나로 학습했습니다. 출처는 플래그로 구분하고, 입력은 충전 세션(사이클) 하나를 요약한 56개 값입니다. 팀 통합본의 16개에 BatICM 원본 로그에서 직접 뽑은 40개(같은 SOC에서의 충전 전압, SOC당 전압 기울기, 셀 간 전압 편차, 온도)를 더했습니다. 출력은 SOH(0~1)와 90% 예측구간이고, 시드가 다른 5개 모델의 평균입니다.

### 성능

처음 보는 배터리·차량 기준입니다. 같은 배터리는 학습과 평가에 섞이지 않게 나눴고, 하이퍼파라미터도 평가 데이터를 보지 않고 골랐습니다(중첩 교차검증). R²는 데이터셋마다 따로 계산했습니다.

| 데이터셋 | MAE | RMSE | R² | 중앙값으로 찍었을 때 RMSE |
|---|---|---|---|---|
| **BatICM** (실차 20대) | **0.0105** | 0.0164 | **0.841** | 0.0411 |
| SNL (시험조건 8개) | 0.0127 | 0.0194 | 0.913 | 0.0657 |
| NASA (셀 34개) | 0.0442 | 0.0652 | 0.814 | 0.1516 |

BatICM 기준으로 **SOH를 약 1%p 오차**로 맞춥니다. 90% 예측구간은 BatICM ±2.2%p, SNL ±2.3%p, NASA ±8.7%p입니다. 신뢰구간, 오차 분석, 비교 모델 같은 자세한 결과는 [training/results/report.md](training/results/report.md)에 있습니다.

성능이 오른 과정은 이렇습니다. 통합본의 16개 피처만 쓰면 BatICM MAE가 0.0145였고(개발 데이터 CV), 열화 피처를 넣어서 0.0117, MAE 손실로 0.0112가 됐고, 튜닝과 5-시드 평균을 거친 최종 값이 0.0105입니다.

### 한계

- 실차 데이터가 **BAIC EU500 20대뿐**이라 다른 차종에서는 검증하지 못했습니다. 실험실 데이터(NASA, SNL)만 학습하면 실차를 전혀 맞추지 못했습니다.
- BatICM의 SOH 정답이 충전 전류 적분으로 **추정한 값**이라 세션마다 ±1~2%p 흔들립니다. 지금 오차(1%p)가 이 잡음 수준에 가까워서 더 줄이는 것은 의미가 없습니다. 실제로 용량을 측정한 값과는 비교하지 않았습니다.
- NASA는 정확도가 낮습니다. SOH가 노화뿐 아니라 방전 부하·온도 같은 시험 조건에 따라서도 달라지기 때문입니다.
- BatICM에서 XGBoost와 RandomForest의 성능은 같았습니다(MAE 0.0105 / 0.0104). XGBoost를 쓴 이유는 결측 처리와 단조 제약, MAE 손실을 쓸 수 있어서입니다.
- 따로 떼어 둔 test는 BatICM 3대, NASA 5개뿐이라 값이 많이 흔들립니다. 그래서 위 표는 76개를 모두 쓴 중첩 CV 기준입니다.

### 모델 사용

```python
import sys; sys.path.insert(0, "training")
from predict_soh import load_model, predict_soh

out = predict_soh(df, dataset="BatICM", model=load_model())  # soh_pred, soh_lo90, soh_hi90
```

`df`에는 `training/results/model_meta.json`의 `features` 56개 컬럼이 있어야 합니다. BatICM 세션 피처는 `training/build_baticm_features.py`와 같은 방식으로 계산합니다.

## 실행 방법

```bash
pip install -r training/requirements.txt

python preprocessing/prepare_training_data.py   # NASA / BatICM 전처리
python preprocessing/prepare_snl_data.py        # SNL 전처리 (입력: datasets/SNL_NMC/raw/snl_capacity_per_cycle.csv)

# 학습 입력은 저장소 루트의 '모든 데이터 병합.xlsx'입니다 (저장소에 포함)
python training/build_baticm_features.py        # BatICM 원본 로그(약 1.1GB, datasets/BatICM/raw_csv)에서 세션 피처 생성
python training/train_final.py --trials 25 --inner-trials 10   # 학습·검증 (약 45분)
python training/compare_rf.py                   # (선택) RandomForest 비교
python training/make_report.py                  # training/results/report.md 생성
python training/tests/test_core.py              # 빠른 테스트
```

전처리에 쓰는 원본 CSV는 저장소에 없고 `datasets/` 아래에 두어야 합니다. SNL은 batteryarchive.org에 다운로드 버튼이 없어서 차트 데이터를 셀별로 직접 추출한 파일을 씁니다. 시드(`42`)와 XGBoost 스레드 수(4)를 고정해서 다른 컴퓨터에서도 같은 결과가 나옵니다. 스레드 수가 달라지면 같은 설정에서도 조기종료 라운드가 크게 달라지는 것을 확인했습니다.

## 저장소 구조

```
preprocessing/   데이터셋별 전처리
training/        피처 생성, 학습·검증, 추론(predict_soh.py), 테스트
  results/       최종 모델(5개), 설정, 지표, 리포트, 그림
datasets/        (git 제외) 원본·전처리 데이터
모든 데이터 병합.xlsx   학습용 통합 데이터 (NASA + BatICM + SNL)
```

## 팀

계명대학교 미래자동차공학전공 4인 팀 (계명대 3명 + 아주대 학점교류 1명) — 기계공학과·교통시스템공학과·전자공학과·산업공학과
