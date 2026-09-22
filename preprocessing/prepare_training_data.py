"""
NASA PCoE, BatICM 데이터를 학습 가능한 형태로 정제한다.

NASA PCoE: type == 'discharge' 이면서 soh가 결측이 아닌 행만 사용한다.
    (charge 사이클 행에는 용량/soh 지표가 원래 없음)
BatICM: soh가 0.5 ~ 1.05 범위를 벗어나는 행(완충 안 된 짧은 세션 등으로 인한
    이상치)을 제거한다.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(r"E:\7-4. 글로벌공학설계프로젝트\프로젝트\datasets")

NASA_SRC = ROOT / "NASA_PCoE_Battery" / "processed" / "nasa_battery_cycle_level.csv"
NASA_OUT = ROOT / "NASA_PCoE_Battery" / "processed" / "nasa_battery_training.csv"

BATICM_SRC = ROOT / "BatICM" / "processed" / "batICM_capacity_per_cycle.csv"
BATICM_OUT = ROOT / "BatICM" / "processed" / "baticm_training.csv"

SOH_MIN, SOH_MAX = 0.5, 1.05


def prepare_nasa():
    df = pd.read_csv(NASA_SRC)
    before = len(df)
    df = df[(df["type"] == "discharge") & df["soh"].notna()].copy()
    df.to_csv(NASA_OUT, index=False)
    print(f"[NASA] {before} -> {len(df)} rows (discharge & soh not null)")
    print(f"[NASA] batteries: {df['battery_id'].nunique()}")
    print(f"[NASA] saved to {NASA_OUT}")


def prepare_baticm():
    df = pd.read_csv(BATICM_SRC)
    before = len(df)
    outliers = df[(df["soh"] < SOH_MIN) | (df["soh"] > SOH_MAX)]
    df = df[(df["soh"] >= SOH_MIN) & (df["soh"] <= SOH_MAX)].copy()
    df.to_csv(BATICM_OUT, index=False)
    print(f"[BatICM] {before} -> {len(df)} rows (soh in [{SOH_MIN}, {SOH_MAX}])")
    print(f"[BatICM] removed {len(outliers)} outlier rows")
    print(f"[BatICM] vehicles: {df['vehicle_id'].nunique()}")
    print(f"[BatICM] saved to {BATICM_OUT}")


if __name__ == "__main__":
    prepare_nasa()
    prepare_baticm()
