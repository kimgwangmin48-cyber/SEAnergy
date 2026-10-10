"""
SEAnergy SOH 학습 공용 모듈: 데이터 로드·정제·무결성 검사, 가중치, 평가 지표, 베이스라인.

설계 원칙
  1. 같은 배터리/차량/셀(엔티티)의 행은 절대 학습과 평가에 나뉘어 들어가지 않는다.
     SNL은 복제 셀(a/b/c)이 같은 시험조건이라 '시험조건' 단위로 묶는다 (group 컬럼).
  2. 정답 계산에 쓰인 capacity_ah_unified, 식별자(row_id/entity_id), split은 입력(X)에서 제외하고 assert로 강제한다.
  3. 전체 R^2는 소스 간 SOH 평균 차이(NASA 0.72 / SNL 0.81 / BatICM 0.84)로 부풀려지므로,
     판단은 '소스별, 엔티티 균등(macro) RMSE'와 '소스별 베이스라인 대비 상대점수(rel_score)'로 한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]  # 저장소 루트 (training/ 의 상위 폴더)
DATA_XLSX = ROOT / "모든 데이터 병합.xlsx"
OUT = ROOT / "training" / "outputs"  # 캐시/중간 산출물 (git 제외)

TARGET = "soh_target_frac"
# 팀 통합본 '칼럼설명' 시트의 X 입력 칼럼 (is_* 는 source_dataset 인코딩)
FEATURES = [
    "cycle_index", "time_elapsed_days", "temperature_c",
    "soc_start_frac", "soc_end_frac", "voltage_mean_v", "current_mean_a",
    "charge_c_rate", "discharge_c_rate",
    "voltage_available", "current_available", "soc_available",
    "rated_capacity_ah", "is_nasa", "is_baticm", "is_snl",
]
FORBIDDEN_IN_X = {"capacity_ah_unified", "row_id", "entity_id", "split", "source_dataset", TARGET, "group", "source"}

# 정제 규칙: 발표자료(슬라이드 9)에 적힌 BatICM 규칙(0.5~1.05)을 통합본에도 동일 적용 + 물리적 상한
CLEAN_RULES = {
    "BatICM": (0.50, 1.05),   # 발표자료 명시 규칙
    "NASA": (0.00, 1.10),     # 하한은 EOL까지 가는 실험이라 유지, 상한만 컷
    "SNL": (0.00, 1.10),
}

# XGBoost 스레드 수는 '컴퓨터와 무관하게' 고정한다. 스레드 수가 바뀌면 부동소수 합산 순서가 달라져
# (특히 MAE 손실은 검증 곡선이 평평해서) 조기종료 라운드와 예측이 달라지므로, 재현성을 위해 4로 고정한다.
NTHREAD = 4

BASE_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
    learning_rate=0.05, max_depth=6, min_child_weight=1.0, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
    seed=SEED, nthread=NTHREAD,
)


# ----------------------------------------------------------------------------- 데이터
def load_data():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "_data_cache.pkl"
    if cache.exists() and cache.stat().st_mtime > DATA_XLSX.stat().st_mtime:
        df = pd.read_pickle(cache)
    else:
        print(f"reading {DATA_XLSX.name} ...", flush=True)
        df = pd.read_excel(DATA_XLSX, sheet_name="SOH학습")
        df.to_pickle(cache)
    df = df.copy()
    df["is_nasa"] = (df["source_dataset"] == "NASA").astype(int)
    df["is_baticm"] = (df["source_dataset"] == "BatICM").astype(int)
    df["is_snl"] = (df["source_dataset"] == "SNL").astype(int)
    df["source"] = df["source_dataset"]
    # SNL 복제 셀(a/b/c)은 같은 시험조건 -> 조건 단위로 그룹화 (복제 셀 누수 방지)
    df["group"] = np.where(df["source"] == "SNL",
                           df["entity_id"].str.replace(r"_[a-z]$", "", regex=True),
                           df["entity_id"])
    return df.reset_index(drop=True)


def clean(df):
    report = {}
    keep = pd.Series(True, index=df.index)
    for src, (lo, hi) in CLEAN_RULES.items():
        m = df["source"] == src
        bad = m & ((df[TARGET] < lo) | (df[TARGET] > hi))
        report[src] = {"rule": f"{lo}<=soh<={hi}", "removed": int(bad.sum()), "of": int(m.sum())}
        keep &= ~bad
    out = df[keep].reset_index(drop=True)
    assert np.isfinite(out[TARGET]).all(), "target has NaN/inf"
    return out, report


def sanity_checks(df, dev, test):
    assert not (set(FEATURES) & FORBIDDEN_IN_X), "forbidden column in FEATURES"
    assert set(dev["entity_id"]).isdisjoint(set(test["entity_id"])), "ENTITY LEAK dev/test"
    assert df["row_id"].is_unique, "duplicate row_id"
    assert set(df["split"].unique()) <= {"train", "validation", "test"}
    # soh == capacity / rated (정답 정의 일관성)
    diff = (df["capacity_ah_unified"] / df["rated_capacity_ah"] - df[TARGET]).abs().max()
    assert diff < 1e-6, f"target definition mismatch {diff}"
    sib = sorted(set(test.loc[test.source == "SNL", "group"]) & set(dev.loc[dev.source == "SNL", "group"]))
    return {"snl_test_conditions_also_in_dev": sib}


# ----------------------------------------------------------------------------- 가중치/지표
def make_weights(d, mode):
    """none | entity (배터리 1개 = 1표) | source_balanced (소스 3개 동일 비중 + 소스 내 엔티티 균등)
    | sb:<소스>=<배율>[,...] (source_balanced 후 해당 소스에 배율, 예: "sb:BatICM=2")"""
    n_e = d.groupby("entity_id")["entity_id"].transform("size").to_numpy(float)
    if mode == "none":
        w = np.ones(len(d))
    elif mode == "entity":
        w = 1.0 / n_e
    elif mode == "source_balanced" or mode.startswith("sb:"):
        w_e = pd.Series(1.0 / n_e, index=d.index)
        w = (w_e / w_e.groupby(d["source"]).transform("sum")).to_numpy()
        if mode.startswith("sb:"):
            for kv in mode[3:].split(","):
                s, k = kv.split("=")
                w = np.where((d["source"] == s).to_numpy(), w * float(k), w)
    else:
        raise ValueError(mode)
    return w / w.mean()


def entity_mse(d, p):
    e = (d[TARGET].to_numpy() - p) ** 2
    return pd.Series(e).groupby(d["entity_id"].to_numpy()).mean()


def evaluate(d, p):
    """소스별 + 엔티티 균등 지표. source_macro_rmse가 소스 간 비교 기준."""
    y = d[TARGET].to_numpy()
    em = entity_mse(d, p)
    ent_src = d.groupby("entity_id")["source"].first()
    per_src = {}
    for s in sorted(d["source"].unique()):
        m = (d["source"] == s).to_numpy()
        ys, ps = y[m], p[m]
        sst = ((ys - ys.mean()) ** 2).sum()
        per_src[s] = {
            "entity_macro_rmse": float(np.sqrt(em[ent_src[ent_src == s].index].mean())),
            "row_rmse": float(np.sqrt(np.mean((ys - ps) ** 2))),
            "mae": float(np.mean(np.abs(ys - ps))),
            "r2_within_source": float(1 - ((ys - ps) ** 2).sum() / sst) if sst > 0 else float("nan"),
            "n_rows": int(m.sum()), "n_entities": int(ent_src[ent_src == s].shape[0]),
        }
    return {"per_source": per_src,
            "source_macro_rmse": float(np.mean([v["entity_macro_rmse"] for v in per_src.values()]))}


def rel_score(dev, oofs, base_oofs):
    """모델 선택 기준: 소스별 (엔티티 균등 RMSE / 같은 fold의 소스중앙값 베이스라인 RMSE)의 평균.
    1 미만이면 베이스라인보다 낫다. NASA처럼 절대 오차가 큰 소스가 선택을 지배하지 않게 한다."""
    sc = []
    for o, b in zip(oofs, base_oofs):
        e, eb = evaluate(dev, o)["per_source"], evaluate(dev, b)["per_source"]
        sc.append(np.mean([e[s]["entity_macro_rmse"] / eb[s]["entity_macro_rmse"] for s in e]))
    return float(np.mean(sc))


# ----------------------------------------------------------------------------- 베이스라인
class SourceMedian:
    """소스별 SOH 중앙값 (이걸 못 넘으면 모델이 배운 게 없다는 뜻)"""
    def fit(self, d):
        self.m = d.groupby("source")[TARGET].median().to_dict()
        return self

    def predict(self, d):
        return d["source"].map(self.m).to_numpy(float)


class SourceLinear:
    """소스별 cycle_index 선형추세 (단순 열화 추세 대비 모델이 더 나은지 확인)"""
    def fit(self, d):
        self.c = {s: np.polyfit(g["cycle_index"].to_numpy(float), g[TARGET].to_numpy(), 1) for s, g in d.groupby("source")}
        return self

    def predict(self, d):
        out = np.zeros(len(d))
        for s, c in self.c.items():
            m = (d["source"] == s).to_numpy()
            out[m] = np.polyval(c, d.loc[m, "cycle_index"].to_numpy(float))
        return out
