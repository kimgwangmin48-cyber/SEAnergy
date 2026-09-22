"""
Sandia National Labs (SNL) 18650 NMC 데이터 후처리.

배터리아카이브 Redash 대시보드에서 브라우저 자동화로 추출한
셀별 사이클 데이터(ah_c, ah_d, e_c, e_d, ah_eff, e_eff)에
soh(ah_d / rated_capacity_ah)와 온도/DoD/C-rate 메타 컬럼을 추가해
학습용 파일로 저장한다.

스케일링(StandardScaler)까지 적용해서 `_scaled` 컬럼을 추가로 만든다.
트리/앙상블 모델(XGBoost 등)은 스케일링이 필요 없지만, 다른 알고리즘을
같이 시도하거나 피처 간 크기를 비교할 때 쓸 수 있게 원본 값과 스케일된
값을 둘 다 남겨둔다. 여기서는 전체 데이터에 대해 스케일러를 학습시켰는데,
실제 모델링 단계에서 train/test를 나누면 스케일러는 train 셋에만
다시 fit하는 걸 권장한다 (여기 스케일러를 그대로 쓰면 test 정보가
train에 살짝 새는 data leakage가 생길 수 있음).
"""

import re
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SRC = Path(r"C:\tmp\snl_capacity_per_cycle.csv")
OUT_DIR = Path(r"E:\7-4. 글로벌공학설계프로젝트\프로젝트\datasets\SNL_NMC\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "snl_nmc_training.csv"
SCALER_OUT = OUT_DIR / "snl_nmc_scaler.joblib"

# 스케일링 대상: 값의 스케일이 서로 다른 수치형 피처만 (id·범주형·타깃(soh) 제외)
SCALE_COLS = [
    "cycle_index",
    "ah_c",
    "ah_d",
    "e_c",
    "e_d",
    "ah_eff",
    "e_eff",
    "temperature_c",
    "soc_min",
    "soc_max",
    "crate_charge",
    "crate_discharge",
    "bol_capacity_ah",
]

RATED_CAPACITY_AH = 3.0  # 참고용(공칭 스펙). 0-100% DoD 셀에만 대략 들어맞음.
# 초반 몇 사이클은 formation/특성평가 구간이라 목표 DoD보다 용량이 크게 찍힌다
# (예: 40-60% 셀도 처음 3~4사이클은 풀레인지로 찍힘). 10~30번째 사이클의
# 중앙값을 BOL(100%) 기준으로 삼아 이 구간을 건너뛴다.
BASELINE_SKIP = 10
BASELINE_WINDOW = 20

CELL_ID_RE = re.compile(
    r"SNL_18650_NMC_(?P<temp>-?\d+)C_(?P<soc_min>\d+(?:\.\d+)?)-(?P<soc_max>\d+(?:\.\d+)?)_"
    r"(?P<crate_c>\d+(?:\.\d+)?)/(?P<crate_d>\d+(?:\.\d+)?)C_(?P<replicate>[a-z])"
)


def parse_cell_id(cell_id):
    m = CELL_ID_RE.match(cell_id)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "temperature_c": float(d["temp"]),
        "soc_min": float(d["soc_min"]),
        "soc_max": float(d["soc_max"]),
        "crate_charge": float(d["crate_c"]),
        "crate_discharge": float(d["crate_d"]),
        "replicate": d["replicate"],
    }


def main():
    df = pd.read_csv(SRC)
    meta = df["cell_id"].apply(parse_cell_id).apply(pd.Series)
    df = pd.concat([df, meta], axis=1)
    df["rated_capacity_ah"] = RATED_CAPACITY_AH

    # DoD 구간이 셀마다 달라(0-100%, 20-80%, 40-60%) 공칭용량 대비 SOH를 쓰면
    # 부분 DoD 셀이 전부 저용량으로 오인된다. 셀별 초기 N사이클 평균 용량을
    # BOL(100%) 기준으로 삼아 상대 SOH를 계산한다.
    before = len(df)
    df = df[df["ah_d"] > 0.05].copy()  # 0에 가까운 결측/글리치 행 제거
    df = df.sort_values(["cell_id", "cycle_index"])

    def robust_baseline(s):
        s = s.reset_index(drop=True)
        if len(s) >= BASELINE_SKIP + BASELINE_WINDOW:
            window = s.iloc[BASELINE_SKIP : BASELINE_SKIP + BASELINE_WINDOW]
        else:
            window = s.iloc[min(5, len(s) - 1) :]
        return window.median()

    baseline = df.groupby("cell_id")["ah_d"].apply(robust_baseline)
    df["bol_capacity_ah"] = df["cell_id"].map(baseline)
    df["soh"] = df["ah_d"] / df["bol_capacity_ah"]

    df = df[(df["soh"] >= 0.5) & (df["soh"] <= 1.15)].copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[SCALE_COLS])
    for i, col in enumerate(SCALE_COLS):
        df[f"{col}_scaled"] = scaled[:, i]
    joblib.dump(scaler, SCALER_OUT)

    df.to_csv(OUT, index=False)
    print(f"rows: {before} -> {len(df)}")
    print(f"cells: {df['cell_id'].nunique()}")
    print(f"scaled columns: {SCALE_COLS}")
    print(f"saved to {OUT}")
    print(f"scaler saved to {SCALER_OUT}")


if __name__ == "__main__":
    main()
