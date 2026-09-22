# SEAnergy

머신러닝 기반 전기차 배터리 SOH(State of Health)·잔존수명 진단 플랫폼

계명대학교 미래자동차공학전공 종합설계 프로젝트로 시작된 프로젝트입니다. 전기차 배터리의 충방전 이력 데이터를 학습해 배터리의 건강 상태(SOH)와 잔존 수명(RUL)을 예측하고, 이를 웹 대시보드로 시각화해서 보여주는 것을 목표로 합니다.

## 배경

전기차 보급이 늘면서 중고 전기차 거래·배터리 진단에 대한 수요도 함께 커지고 있습니다. 하지만 기존 OBD2 기반 진단은 단순 에러 코드 조회 수준에 그쳐, 배터리가 실제로 얼마나 열화됐는지 정밀하게 판단하기 어렵습니다. 이 프로젝트는 여러 공개 배터리 열화 데이터셋을 학습한 머신러닝 모델로 이 문제를 보완하려는 시도입니다.

## 현재 진행 상황

이번 학기(2026-2) 범위는 데이터 수집·전처리 → AI 모델링 → API → 웹사이트로 이어지는 파이프라인 구축이며, 현재는 **데이터 수집·전처리 단계**까지 진행되어 있습니다. 실차 OBD2 하드웨어 연동은 겨울 Bellevue College와의 글로벌 공동 프로젝트로 이어질 예정입니다.

## 데이터셋

| 데이터셋 | 화학종 | 성격 | 상태 |
|---|---|---|---|
| [NASA PCoE](https://www.kaggle.com/datasets/ckskaggle/li-ion-battery-dataset-from-nasa-pcoe) | LCO | 실험실 가속열화 | 채택 |
| [BatICM](https://github.com/BatICM/battery-charging-data-of-on-road-electric-vehicles) | NCM | 실차 20대, 29개월 실사용 데이터 | 채택 (메인) |
| [SNL (batteryarchive.org)](http://batteryarchive.org) | NMC | 온도×방전심도×C-rate 통제 실험 | 검토 중 |

원본·전처리 데이터는 용량 문제로(2GB+) 이 저장소에 포함하지 않습니다. `preprocessing/`의 스크립트로 각자 재생성해서 사용하세요.

## 실행 방법

```bash
pip install pandas

# NASA PCoE / BatICM 전처리
python preprocessing/prepare_training_data.py

# SNL NMC 전처리
python preprocessing/prepare_snl_data.py
```

각 스크립트는 원본 CSV(전처리 전 데이터)가 `datasets/<데이터셋명>/processed/` 아래 있다고 가정하고, 결측치·이상치를 제거한 학습용 CSV를 같은 경로에 생성합니다.

## 저장소 구조

```
preprocessing/   # 데이터셋별 전처리 스크립트
datasets/        # (git 제외) 원본/전처리 데이터
```

## 팀

계명대학교 미래자동차공학전공 4인 팀 (계명대 3명 + 아주대 학점교류 1명) — 기계공학과·교통시스템공학과·전자공학과·산업공학과 융합
