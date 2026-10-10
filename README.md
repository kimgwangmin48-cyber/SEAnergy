# SEAnergy

머신러닝 기반 전기차 배터리 SOH(State of Health)·잔존수명 진단 플랫폼

계명대학교 미래자동차공학전공 종합설계 프로젝트로 시작된 프로젝트입니다. 전기차 배터리의 충방전 이력 데이터를 학습해 배터리의 건강 상태(SOH)와 잔존 수명(RUL)을 예측하고, 이를 웹 대시보드로 시각화해서 보여주는 것을 목표로 합니다.

## 배경

전기차 보급이 늘면서 중고 전기차 거래·배터리 진단에 대한 수요도 함께 커지고 있습니다. 하지만 기존 OBD2 기반 진단은 단순 에러 코드 조회 수준에 그쳐, 배터리가 실제로 얼마나 열화됐는지 정밀하게 판단하기 어렵습니다. 이 프로젝트는 여러 공개 배터리 열화 데이터셋을 학습한 머신러닝 모델로 이 문제를 보완하려는 시도입니다.

## 현재 진행 상황

| 단계 | 상태 |
|---|---|
| 데이터 수집·전처리 (NASA, BatICM, SNL) | 완료 |
| SOH 예측 모델 (XGBoost) 학습·검증 | **완료** — [아래 결과](#soh-예측-모델) |
| 모델 API · 웹 대시보드 | 진행 예정 |
| 실차 OBD2 하드웨어 연동 | 겨울 Bellevue College 글로벌 공동 프로젝트로 이어질 예정 |

## 데이터셋

| 데이터셋 | 화학종 | 성격 | 상태 |
|---|---|---|---|
| [NASA PCoE](https://www.kaggle.com/datasets/ckskaggle/li-ion-battery-dataset-from-nasa-pcoe) | LCO | 실험실 가속열화 (34개 셀) | 채택 |
| [BatICM](https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles) | NCM | 실차 20대(BAIC EU500, CATL 145Ah 팩), 29개월 실사용 충전 로그 | 채택 (메인) |
| [SNL (batteryarchive.org)](http://batteryarchive.org) | NMC | 온도×방전심도×C-rate 통제 실험 (22개 셀, 시험조건 8가지) | 채택 |

원본·전처리 데이터와 통합 학습 데이터(`모든 데이터 병합.xlsx`)는 용량·출처 라이선스 문제로 이 저장소에 포함하지 않습니다. BatICM 데이터를 쓸 때는 원 논문(Applied Energy, 2023)을 인용하세요.

## SOH 예측 모델

세 데이터셋을 한 모델로 학습한 **XGBoost 회귀 모델**입니다. 충전 세션(또는 사이클) 하나의 측정 요약을 입력하면 그 시점의 SOH(0~1)와 90% 예측구간을 냅니다. 최종 예측은 시드가 다른 5개 모델의 평균입니다.

### 성능 (처음 보는 배터리/차량 기준)

바깥 5-fold로 평가하고 하이퍼파라미터는 각 fold 안쪽에서만 고른 **중첩 교차검증**입니다. 같은 배터리·차량(SNL은 같은 시험조건)은 항상 같은 fold에 들어가므로, 학습에 쓴 배터리가 평가에 섞이지 않습니다. 괄호는 95% 신뢰구간(배터리/차량 단위 부트스트랩)입니다.

| 데이터셋 | 검증 단위 | MAE | RMSE | R² | (참고) 중앙값 예측 RMSE |
|---|---|---|---|---|---|
| **BatICM** (실차) | 차량 20대 | **0.0105** (0.0097~0.0114) | 0.0164 | **0.841** (0.808~0.869) | 0.0411 |
| SNL (통제 실험) | 시험조건 8개 | 0.0127 (0.0095~0.0165) | 0.0194 | 0.913 (0.842~0.950) | 0.0657 |
| NASA (실험실) | 셀 34개 | 0.0442 (0.0344~0.0573) | 0.0652 | 0.814 (0.709~0.884) | 0.1516 |

- SOH는 0~1 단위입니다. BatICM의 MAE 0.0105는 **SOH 약 1%p 오차**입니다.
- R²는 데이터셋별로만 계산했습니다. 세 데이터셋을 합쳐서 내면 소스마다 SOH 평균이 달라(NASA 0.72 / SNL 0.81 / BatICM 0.84) 값이 부풀려집니다.
- 90% 예측구간은 BatICM ±2.2%p, SNL ±2.3%p, NASA ±8.7%p이고, 교차검증에서 실제로 정답이 구간 안에 든 비율은 각각 0.90 / 0.90 / 0.89입니다.
- BatICM 20대 각각의 MAE는 0.0084~0.0157로, 특정 차량에서만 잘 맞는 것이 아닙니다.
- 예측 오차가 데이터 자체의 한계에 가깝습니다. BatICM의 SOH 정답은 세션마다 차량 추세 주위로 MAE 0.0125 정도 흔들리는데, 모델 오차(0.0105)가 이보다도 작습니다.
- 자세한 표(test, 설계 비교, 오차 분석, 피처 중요도)는 [training/results/report.md](training/results/report.md)에 있습니다.

### 무엇이 성능을 올렸나 (개발 데이터 그룹 CV, BatICM)

| 설정 | MAE | RMSE |
|---|---|---|
| 팀 통합본 16개 피처 (기준) | 0.0145 | 0.0206 |
| + BatICM 세션 열화 피처 40개 | 0.0117 | 0.0176 |
| + SOH 단조 제약 (노화에 대해 감소) | 0.0119 | 0.0179 |
| + MAE 손실 (정답에 튀는 값이 있어서) | 0.0112 | 0.0171 |
| + 깊은 트리·서브샘플링 튜닝, 5-시드 평균 | 중첩 CV에서 0.0105 | 중첩 CV에서 0.0164 |

가장 크게 오른 것은 **BatICM 원본 로그에서 만든 세션 열화 피처**입니다. 같은 SOC에서의 충전 전압(`v_at_70` 등), SOC당 전압 기울기(`dVdSOC_at_60_to_90`), 셀 간 전압 편차, 온도와 그 롤링 요약이며, 중요도 상위를 차지합니다. 이 피처가 없으면 모델은 사실상 `cycle_index`·경과일 추세만 배웁니다.

### 검증 방법과 누수 방지

- **정답 누수 차단**: BatICM의 SOH 정답은 "충전 전류 적분 ÷ SOC 변화량"(쿨롱 카운팅)으로 만든 값입니다. 그래서 전류×시간으로 정답이 복원되는 입력(세션 지속시간, 샘플 수, 시간 기준 전압 기울기)과 BMS가 이미 계산한 `available_capacity`·`available_energy`는 입력에서 뺐고, 코드에서 검사합니다.
- **인과적 롤링**: 롤링 요약은 현재와 과거 세션만 씁니다. 미래 정보를 쓰지 않습니다.
- **정답 셔플 검사**: 정답을 섞어 학습하면 test 성능이 중앙값 예측 수준으로 떨어집니다(누수가 있으면 점수가 좋게 나옵니다).
- **설계·튜닝은 개발 데이터에서만**, 따로 둔 test(11개 엔티티)는 마지막에 한 번만 평가했습니다.
- **비교 기준선**: 데이터셋별 중앙값 예측, 데이터셋별 cycle 선형 추세, RandomForest, 초기 모델(팀 통합본 16피처).

### 한계 (알고 쓰셔야 하는 점)

- **실차 데이터는 BAIC EU500 20대뿐입니다.** 다른 차종·배터리에는 검증하지 않았습니다. 초기 실험에서 실차(BatICM)를 빼고 NASA·SNL만으로 학습하면 실차를 전혀 못 맞췄습니다(R² 약 -18). 실험실 데이터가 실차를 대신하지 못합니다.
- **NASA는 정확도가 낮습니다.** NASA의 SOH 정답이 노화뿐 아니라 시험 프로토콜(방전 부하 1/2/4A, 주변 온도 4/24/43°C)에 따라 달라지기 때문입니다.
- **SNL은 시험조건이 8가지뿐**이라 신뢰구간이 넓습니다.
- **BatICM에서는 XGBoost와 RandomForest가 성능이 같습니다.** 같은 fold에서 MAE 0.0105 vs 0.0104입니다. SNL(MAE 0.0127 vs 0.0148)과 NASA(0.0442 vs 0.0481)에서는 XGBoost가 낫지만 신뢰구간이 겹칩니다. XGBoost는 결측 처리, 단조 제약, MAE 손실을 쓸 수 있어 채택했습니다.
- 따로 둔 test는 엔티티가 BatICM 3대·SNL 2조건·NASA 5개뿐이라 흔들림이 큽니다(예: NASA test R² 0.40, BatICM test RMSE 0.0172는 RF 0.0168과 비슷). 보고 수치는 76개 엔티티를 모두 쓴 중첩 CV 기준이고, test 표는 [리포트](training/results/report.md)에 있습니다.
- SOH 정답 자체가 충전 용량으로 추정한 값이라 세션마다 ±1~2%p 잡음이 있습니다. 이 아래로 오차를 줄이는 것은 의미가 없습니다.

### 시도했지만 채택하지 않은 것

개발 데이터 CV에서 효과가 없었거나 나빠져서 뺐습니다.

- 사용 이력 피처 (방치 시간, 최근 30일 세션 수, 누적 급속충전·고온 비율), 내부저항 대용치 → 이득 없음 (BatICM이 이미 정답 잡음 수준에 가까움)
- NASA 시험조건(주변 온도·전류·컷오프 전압) 피처 → 오히려 악화
- 학습 정답 평활, Pseudo-Huber 손실 → 개선 없음 / 수치 불안정
- 데이터셋별 별도 모델 → BatICM만 약 4% 좋아지고 SNL·NASA는 나빠져서 통합 모델 유지

### 모델 사용

```python
import sys; sys.path.insert(0, "training")
from predict_soh import load_model, predict_soh

model = load_model()                                   # training/results/ 의 5개 모델 + 설정
out = predict_soh(df, dataset="BatICM", model=model)   # soh_pred, soh_lo90, soh_hi90
```

`df`는 `training/results/model_meta.json`의 `features` 56개 컬럼을 가져야 합니다(없는 측정값은 NaN). BatICM 세션 피처는 `training/build_baticm_features.py`와 같은 방식으로 계산합니다.

## 실행 방법

```bash
pip install -r training/requirements.txt

# 1) 데이터 전처리 (원본은 datasets/ 아래에 둡니다)
python preprocessing/prepare_training_data.py   # NASA PCoE / BatICM
python preprocessing/prepare_snl_data.py        # SNL NMC (스케일링 포함)

# 2) 모델 학습 (저장소 루트에 '모든 데이터 병합.xlsx'가 있어야 합니다)
python training/build_baticm_features.py        # BatICM 원본 로그 -> 세션 열화 피처 (원본 1.1GB 필요)
python training/train_final.py --trials 25 --inner-trials 10  # 설계 선택 → 튜닝 → test → 중첩 CV → 최종 모델 (약 45분, CPU 20스레드 기준)
python training/compare_rf.py                   # (선택) RandomForest 비교
python training/make_report.py                  # training/results/report.md 생성

python training/tests/test_core.py              # 핵심 로직 빠른 테스트 (합성 데이터, 데이터셋 불필요, 수 초)
```

전처리 스크립트는 원본 CSV가 `datasets/<데이터셋명>/processed/` 아래 있다고 가정하고, 결측치·이상치를 제거한 학습용 CSV를 같은 경로에 생성합니다.

**재현성**: 시드를 고정(`seed=42`)했고, XGBoost 스레드 수도 4로 고정했습니다. 스레드 수가 달라지면 부동소수 합산 순서가 바뀌어 조기종료 라운드와 예측이 달라지기 때문입니다(같은 설정인데 라운드가 73에서 1166까지 달라진 것을 확인). 사용한 라이브러리 버전은 `training/requirements.txt`에 있습니다. fold 병렬 학습 정도(`SOH_N_JOBS`)는 결과에 영향이 없고 테스트로 검증합니다.

## 저장소 구조

```
preprocessing/            # 데이터셋별 전처리 스크립트
training/
  build_baticm_features.py   # BatICM 세션 열화 피처 생성 (정답 재현 검증 포함)
  soh_common.py              # 공용 함수 (데이터 로드·정제·무결성 검사·가중치·지표·베이스라인)
  train_final.py             # 최종 학습·검증 파이프라인 (단계별 함수, fold 병렬 + DMatrix 캐시)
  compare_rf.py, make_report.py, predict_soh.py
  tests/test_core.py         # 핵심 로직 테스트
  results/                   # 최종 모델(5개 시드), 설정, 지표(metrics.json), 리포트, 그림
datasets/                 # (git 제외) 원본/전처리 데이터
```

## 팀

계명대학교 미래자동차공학전공 4인 팀 (계명대 3명 + 아주대 학점교류 1명) — 기계공학과·교통시스템공학과·전자공학과·산업공학과 융합
