"""
SEAnergy SOH 예측 -- XGBoost 학습 파이프라인 (그룹 단위 검증 · 누수 방지 중심)

입력: 프로젝트 루트의 `모든 데이터 병합.xlsx` (팀 통합본, 71,080행)
출력: training/outputs/ (모델, 지표 JSON, 리포트, 그림, OOF 예측)

설계 원칙 (치명적 오류 방지)
  1. 같은 배터리/차량/셀(엔티티)의 행은 절대 train/validation/test에 나뉘어 들어가지 않는다.
     SNL은 복제 셀(a/b/c)이 같은 시험조건이라 '조건' 단위로 묶어서 CV한다.
  2. 정답 계산에 쓰인 capacity_ah_unified, 식별자(row_id/entity_id), split은 입력(X)에서 제외하고 assert로 강제한다.
  3. test는 마지막에 1회만 쓴다. 하이퍼파라미터/가중치 방식 선택은 dev(train+validation)의 그룹 CV로만 한다.
  4. 전체 R^2는 소스 간 SOH 차이(NASA 0.72 / SNL 0.81 / BatICM 0.84)로 부풀려지므로,
     판단은 '소스별, 엔티티 균등(macro) RMSE'와 '소스별 베이스라인 대비 개선'으로 한다.
  5. 셔플 정답(sanity) 검사, 소스 간 일반화(LOSO), 엔티티 단위 부트스트랩 신뢰구간으로 결과를 검증한다.

실행: python train_soh_xgb.py [--trials 40] [--quick]
"""

import argparse
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]  # 저장소 루트 (training/ 의 상위 폴더)
DATA_XLSX = ROOT / "모든 데이터 병합.xlsx"
OUT = ROOT / "training" / "outputs"

TARGET = "soh_target_frac"
# 팀 통합본 '칼럼설명' 시트의 X 입력 칼럼 (is_* 는 source_dataset 인코딩)
FEATURES = [
    "cycle_index", "time_elapsed_days", "temperature_c",
    "soc_start_frac", "soc_end_frac", "voltage_mean_v", "current_mean_a",
    "charge_c_rate", "discharge_c_rate",
    "voltage_available", "current_available", "soc_available",
    "rated_capacity_ah", "is_nasa", "is_baticm", "is_snl",
]
FORBIDDEN_IN_X = {"capacity_ah_unified", "row_id", "entity_id", "split", "source_dataset",
                  TARGET, "group", "source"}
SOURCE_ID_FEATURES = ["rated_capacity_ah", "is_nasa", "is_baticm", "is_snl"]  # LOSO에서 제외

# 정제 규칙: 발표자료(슬라이드 9)에 적힌 BatICM 규칙(0.5~1.05)을 통합본에도 동일 적용 + 물리적 상한
CLEAN_RULES = {
    "BatICM": (0.50, 1.05),   # 발표자료 명시 규칙
    "NASA": (0.00, 1.10),     # 하한은 EOL까지 가는 실험이라 유지, 상한만 컷
    "SNL": (0.00, 1.10),
}

NTHREAD = max(1, (os.cpu_count() or 4) - 1)


def log(msg=""):
    print(msg, flush=True)


# ----------------------------------------------------------------------------- 데이터
def load_data():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "_data_cache.pkl"
    if cache.exists() and cache.stat().st_mtime > DATA_XLSX.stat().st_mtime:
        df = pd.read_pickle(cache)
    else:
        log(f"reading {DATA_XLSX.name} ...")
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
    """none | entity (배터리 1개 = 1표) | source_balanced (소스 3개 동일 비중 + 소스 내 엔티티 균등)"""
    n_e = d.groupby("entity_id")["entity_id"].transform("size").to_numpy(float)
    if mode == "none":
        w = np.ones(len(d))
    elif mode == "entity":
        w = 1.0 / n_e
    elif mode == "source_balanced" or mode.startswith("sb:"):
        w_e = pd.Series(1.0 / n_e, index=d.index)
        w = (w_e / w_e.groupby(d["source"]).transform("sum")).to_numpy()
        if mode.startswith("sb:"):  # 예: "sb:BatICM=2" -> 소스 균등 가중 후 BatICM(배포 대상 도메인)에 2배
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
    """소스별 + 엔티티 균등 지표. source_macro_rmse가 모델 선택 기준."""
    y = d[TARGET].to_numpy()
    em = entity_mse(d, p)
    ent_src = d.groupby("entity_id")["source"].first()
    res, per_src = {}, {}
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
    res["per_source"] = per_src
    res["source_macro_rmse"] = float(np.mean([v["entity_macro_rmse"] for v in per_src.values()]))
    res["overall_row_rmse"] = float(np.sqrt(np.mean((y - p) ** 2)))
    res["overall_r2_(source gap inflated)"] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return res


# ----------------------------------------------------------------------------- 베이스라인
class SourceMedian:
    """소스별 SOH 중앙값 (이걸 못 넘으면 모델이 배운 게 없다는 뜻)"""
    def fit(self, d):
        self.m = d.groupby("source")[TARGET].median().to_dict()
        return self

    def predict(self, d):
        return d["source"].map(self.m).to_numpy(float)


class SourceLinear:
    """소스별 cycle_index 선형추세 (단순 열화 추세 대비 트리모델이 더 나은지 확인)"""
    def fit(self, d):
        self.c = {}
        for s, g in d.groupby("source"):
            self.c[s] = np.polyfit(g["cycle_index"].to_numpy(float), g[TARGET].to_numpy(), 1)
        return self

    def predict(self, d):
        out = np.zeros(len(d))
        for s, c in self.c.items():
            m = (d["source"] == s).to_numpy()
            out[m] = np.polyval(c, d.loc[m, "cycle_index"].to_numpy(float))
        return out


# ----------------------------------------------------------------------------- XGBoost
BASE_PARAMS = dict(
    objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
    learning_rate=0.05, max_depth=6, min_child_weight=1.0, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
    seed=SEED, nthread=NTHREAD,
)


def dmat(d, w=None, feats=FEATURES):
    return xgb.DMatrix(d[feats].to_numpy(float), label=d[TARGET].to_numpy(float),
                       weight=w, feature_names=list(feats), missing=np.nan)


def fit_es(params, dtr, dva, max_rounds=4000, early=100):
    bst = xgb.train(params, dtr, num_boost_round=max_rounds, evals=[(dva, "valid")],
                    early_stopping_rounds=early, verbose_eval=False)
    return bst, int(bst.best_iteration) + 1


def predict(bst, d, feats=FEATURES, n_rounds=None):
    dm = xgb.DMatrix(d[feats].to_numpy(float), feature_names=list(feats), missing=np.nan)
    if n_rounds is None:
        return bst.predict(dm)
    return bst.predict(dm, iteration_range=(0, n_rounds))


def cv_xgb(dev, folds, params, mode, train_source=None, eval_source=None):
    """dev 그룹 CV. 조기종료는 fold 검증셋(엔티티 균등 가중)에서 -> OOF 예측/최적 라운드 반환.
    train_source: 학습 행을 해당 소스로 제한 (소스 단독 학습 ablation용)
    eval_source : 조기종료·평가를 해당 소스의 검증행으로만 수행 (ablation에서 공정 비교용)"""
    oof = np.full(len(dev), np.nan)
    iters = []
    src = dev["source"].to_numpy()
    for tr, va in folds:
        if train_source is not None:
            tr = tr[src[tr] == train_source]
        if eval_source is not None:
            va = va[src[va] == eval_source]
        if len(tr) == 0 or len(va) == 0:
            continue
        dtr_df, dva_df = dev.iloc[tr], dev.iloc[va]
        dtr = dmat(dtr_df, make_weights(dtr_df, mode))
        dva = dmat(dva_df, make_weights(dva_df, "entity"))
        bst, n = fit_es(params, dtr, dva)
        oof[va] = predict(bst, dva_df, n_rounds=n)
        iters.append(n)
    check = np.isfinite(oof) if eval_source is None else np.isfinite(oof[src == eval_source])
    assert check.all(), "OOF has missing predictions"
    return oof, iters


def cv_xgb_rep(dev, fold_sets, params, mode):
    """반복 그룹 CV (fold 분할 시드를 바꿔 여러 번). 반복별 OOF 리스트와 라운드 목록 반환."""
    oofs, iters = [], []
    for folds in fold_sets:
        o, it = cv_xgb(dev, folds, params, mode)
        oofs.append(o)
        iters += it
    return oofs, iters


def rel_score(dev, oofs, base_oofs):
    """모델 선택 기준: 소스별 (엔티티 균등 RMSE / 같은 fold의 소스중앙값 베이스라인 RMSE)의 평균.
    1 미만이면 베이스라인보다 낫다. NASA처럼 절대 오차가 큰 소스가 선택을 지배하지 않게 한다."""
    sc = []
    for o, b in zip(oofs, base_oofs):
        e, eb = evaluate(dev, o)["per_source"], evaluate(dev, b)["per_source"]
        sc.append(np.mean([e[s]["entity_macro_rmse"] / eb[s]["entity_macro_rmse"] for s in e]))
    return float(np.mean(sc))


def sample_params(rng):
    return dict(
        learning_rate=float(np.exp(rng.uniform(np.log(0.02), np.log(0.15)))),
        max_depth=int(rng.randint(3, 10)),
        min_child_weight=float(np.exp(rng.uniform(np.log(1), np.log(200)))),
        subsample=float(rng.uniform(0.6, 1.0)),
        colsample_bytree=float(rng.uniform(0.5, 1.0)),
        reg_lambda=float(np.exp(rng.uniform(np.log(0.1), np.log(50)))),
        reg_alpha=float(np.exp(rng.uniform(np.log(1e-3), np.log(5)))),
        gamma=float(rng.choice([0.0, 0.01, 0.1, 1.0])),
    )


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--quick", action="store_true", help="스모크 테스트(트라이얼 3, 부트스트랩 50)")
    args = ap.parse_args()
    if args.quick:
        args.trials = 3
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    R = {"env": {"python": sys.version.split()[0], "xgboost": xgb.__version__, "sklearn": sklearn.__version__,
                 "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(),
                 "nthread": NTHREAD, "seed": SEED}}
    log(f"== env: xgboost {xgb.__version__} / sklearn {sklearn.__version__} / nthread {NTHREAD}")

    # 1) 데이터 ------------------------------------------------------------------------
    raw = load_data()
    df, clean_rep = clean(raw)
    R["cleaning"] = clean_rep
    log(f"== rows {len(raw)} -> {len(df)} after cleaning: {json.dumps(clean_rep, ensure_ascii=False)}")
    dev = df[df["split"].isin(["train", "validation"])].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    R["sanity"] = sanity_checks(df, dev, test)
    R["data"] = {
        "dev_rows": len(dev), "test_rows": len(test),
        "dev_entities": int(dev.entity_id.nunique()), "test_entities": int(test.entity_id.nunique()),
        "dev_groups": int(dev.group.nunique()),
    }
    log(f"== dev {len(dev)} rows / {dev.entity_id.nunique()} entities ({dev.group.nunique()} groups) | "
        f"test {len(test)} rows / {test.entity_id.nunique()} entities")
    log(f"== SNL test 조건이 dev에도 존재(복제셀): {R['sanity']['snl_test_conditions_also_in_dev']} -> SNL test 점수는 낙관적일 수 있음")

    n_rep = 1 if args.quick else 3
    fold_sets = []
    for r in range(n_rep):
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        fs = list(skf.split(dev, dev["source"], groups=dev["group"]))
        for i, (tr, va) in enumerate(fs):
            assert set(dev.iloc[tr].group).isdisjoint(set(dev.iloc[va].group)), "GROUP LEAK in CV"
            if r == 0:
                log(f"   fold{i}: val sources = {dev.iloc[va].groupby('source').group.nunique().to_dict()}")
        fold_sets.append(fs)
    folds = fold_sets[0]
    log(f"   grouped CV: 5 folds x {n_rep} repeats (StratifiedGroupKFold, group=entity / SNL=condition)")

    # 2) 베이스라인 (CV OOF + test) ----------------------------------------------------
    log("\n== [2] baselines")
    base = {}
    base_oofs = {}
    for name, cls in [("source_median", SourceMedian), ("source_linear_cycle", SourceLinear)]:
        oofs = []
        for fs in fold_sets:
            oof = np.zeros(len(dev))
            for tr, va in fs:
                oof[va] = cls().fit(dev.iloc[tr]).predict(dev.iloc[va])
            oofs.append(oof)
        base_oofs[name] = oofs
        base[name] = {"cv": evaluate(dev, oofs[0]),
                      "test": evaluate(test, cls().fit(dev).predict(test))}
        log(f"   {name:22s} CV source-macro RMSE {base[name]['cv']['source_macro_rmse']:.4f} | "
            f"test {base[name]['test']['source_macro_rmse']:.4f}")
    R["baselines"] = base
    bmed = base_oofs["source_median"]

    # 3) 가중치 방식 비교 (기본 파라미터) ------------------------------------------------
    log("\n== [3] weight mode comparison (default params, grouped CV; score = RMSE / source-median baseline, <1 is better)")
    wm = {}
    for mode in ["none", "entity", "source_balanced"]:
        oofs, iters = cv_xgb_rep(dev, fold_sets, BASE_PARAMS, mode)
        wm[mode] = {"cv": evaluate(dev, oofs[0]), "best_iters": iters, "rel_score": rel_score(dev, oofs, bmed)}
        ps = {s: round(v["entity_macro_rmse"], 4) for s, v in wm[mode]["cv"]["per_source"].items()}
        log(f"   {mode:16s} rel-score {wm[mode]['rel_score']:.4f} | per-source RMSE {ps} | iters med {int(np.median(iters))}")
    best_mode = min(wm, key=lambda m: wm[m]["rel_score"])
    R["weight_mode_comparison"] = wm
    R["chosen_weight_mode"] = best_mode
    log(f"   -> chosen weight mode: {best_mode}")

    # 4) 랜덤포레스트 비교 (발표자료: RF vs XGBoost) ---------------------------------------
    log("\n== [4] RandomForest comparison (CV repeat 0 + test)")
    def fit_rf(d):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7,
                                   n_jobs=NTHREAD, random_state=SEED)
        rf.fit(d[FEATURES], d[TARGET], sample_weight=make_weights(d, best_mode))
        return rf
    oof_rf = np.zeros(len(dev))
    for tr, va in folds:
        oof_rf[va] = fit_rf(dev.iloc[tr]).predict(dev.iloc[va][FEATURES])
    rf_full = fit_rf(dev)
    p_rf_test = rf_full.predict(test[FEATURES])
    R["random_forest"] = {"cv": evaluate(dev, oof_rf), "test": evaluate(test, p_rf_test),
                          "cv_rel_score": rel_score(dev, [oof_rf], [bmed[0]])}
    log(f"   RF  CV rel-score {R['random_forest']['cv_rel_score']:.4f} | CV source-macro RMSE "
        f"{R['random_forest']['cv']['source_macro_rmse']:.4f} | test {R['random_forest']['test']['source_macro_rmse']:.4f}")

    # 5) 하이퍼파라미터 랜덤서치 (그룹 CV) --------------------------------------------------
    log(f"\n== [5] random search ({args.trials} trials, grouped CV x{n_rep}, weight={best_mode})")
    rng = np.random.RandomState(SEED)
    trials = []
    best = {"score": wm[best_mode]["rel_score"], "params": dict(BASE_PARAMS),
            "iters": wm[best_mode]["best_iters"]}
    keys = list(sample_params(np.random.RandomState(0)))
    for t in range(args.trials):
        p = dict(BASE_PARAMS)
        p.update(sample_params(rng))
        ts = time.time()
        oofs, iters = cv_xgb_rep(dev, fold_sets, p, best_mode)
        sc = rel_score(dev, oofs, bmed)
        trials.append({"trial": t, "rel_score": sc, "median_iters": int(np.median(iters)), **{k: p[k] for k in keys}})
        mark = ""
        if sc < best["score"]:
            best = {"score": sc, "params": p, "iters": iters}
            mark = "  <-- best"
        log(f"   trial {t:02d} rel-score {sc:.4f} ({time.time()-ts:.0f}s){mark}")
    R["search"] = {"trials": trials, "best_cv_rel_score": best["score"], "default_params_rel_score": wm[best_mode]["rel_score"]}
    pd.DataFrame(trials).to_csv(OUT / "search_trials.csv", index=False)
    best_params = best["params"]
    n_rounds_dev = int(np.ceil(np.median(best["iters"]) * 1.1))
    R["best_params"] = {k: best_params[k] for k in best_params if k not in ("seed", "nthread")}
    R["n_rounds_dev"] = n_rounds_dev
    log(f"   best CV rel-score {best['score']:.4f} (default params {wm[best_mode]['rel_score']:.4f}) | rounds {n_rounds_dev}")

    # best params OOF (최종 CV 성능 + 예측 저장)
    oofs_best, _ = cv_xgb_rep(dev, fold_sets, best_params, best_mode)
    oof_best = oofs_best[0]
    R["xgb_cv_best"] = evaluate(dev, oof_best)
    R["xgb_cv_best"]["rel_score"] = rel_score(dev, oofs_best, bmed)
    pd.DataFrame({"row_id": dev.row_id, "entity_id": dev.entity_id, "source": dev.source,
                  "y": dev[TARGET], "oof_pred": oof_best}).to_csv(OUT / "oof_predictions.csv", index=False)

    # 5b) ablation: 소스 단독 학습 vs 통합 학습 (데이터를 섞는 게 실제로 도움이 되는지) ------------------
    log("\n== [5b] ablation: train on that source only vs. all sources (same folds, same early-stop policy: that source's val rows)")
    abl = {}
    for s in ["NASA", "BatICM", "SNL"]:
        m = (dev["source"] == s).to_numpy()
        oo_only, _ = cv_xgb(dev, folds, best_params, best_mode, train_source=s, eval_source=s)
        oo_all, _ = cv_xgb(dev, folds, best_params, best_mode, eval_source=s)
        only = evaluate(dev[m], oo_only[m])["per_source"][s]["entity_macro_rmse"]
        allv = evaluate(dev[m], oo_all[m])["per_source"][s]["entity_macro_rmse"]
        abl[s] = {"only_this_source_rmse": only, "all_sources_rmse": allv}
        log(f"   {s:7s} only-{s}: {only:.4f} | all-sources: {allv:.4f} -> "
            f"{'mixing HELPS' if allv < only else 'mixing does NOT help'} ({(1-allv/only)*100:+.1f}%)")
    R["ablation_source_only_vs_all"] = abl

    # 6) 최종 학습(dev) -> test 1회 평가 ----------------------------------------------------
    log("\n== [6] final fit on dev -> test (touched once)")
    w_dev = make_weights(dev, best_mode)
    bst = xgb.train(best_params, dmat(dev, w_dev), num_boost_round=n_rounds_dev, verbose_eval=False)
    p_test = predict(bst, test)
    R["xgb_test"] = evaluate(test, p_test)
    R["xgb_test_vs_baseline"] = {}
    log(f"   XGB   test source-macro RMSE {R['xgb_test']['source_macro_rmse']:.4f}")
    for s, v in R["xgb_test"]["per_source"].items():
        b0 = base["source_median"]["test"]["per_source"][s]["entity_macro_rmse"]
        b1 = base["source_linear_cycle"]["test"]["per_source"][s]["entity_macro_rmse"]
        imp = 1 - v["entity_macro_rmse"] / b0
        R["xgb_test_vs_baseline"][s] = {"xgb": v["entity_macro_rmse"], "median_baseline": b0,
                                        "linear_baseline": b1, "improvement_vs_median": imp}
        log(f"     {s:7s} RMSE {v['entity_macro_rmse']:.4f} (median {b0:.4f}, linear {b1:.4f}) "
            f"-> {imp*100:+.1f}% vs median | MAE {v['mae']:.4f} | R2(within) {v['r2_within_source']:.3f}")
    pd.DataFrame({"row_id": test.row_id, "entity_id": test.entity_id, "source": test.source,
                  "cycle_index": test.cycle_index, "y": test[TARGET], "pred": p_test}
                 ).to_csv(OUT / "test_predictions.csv", index=False)
    rf_t = R["random_forest"]["test"]
    rf_detail = ", ".join("%s %.4f" % (s, v["entity_macro_rmse"]) for s, v in rf_t["per_source"].items())
    log(f"   RF    test source-macro RMSE {rf_t['source_macro_rmse']:.4f} ({rf_detail})")

    # 6b) 시드 안정성: subsample/colsample 무작위성 때문에 시드별로 test가 얼마나 흔들리는지 -------------
    log("\n== [6b] seed stability (5 seeds, same params)")
    seed_rows = []
    for sd in [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]:
        pp = dict(best_params, seed=sd)
        b = xgb.train(pp, dmat(dev, w_dev), num_boost_round=n_rounds_dev, verbose_eval=False)
        e = evaluate(test, predict(b, test))
        seed_rows.append({"seed": sd, **{s: v["entity_macro_rmse"] for s, v in e["per_source"].items()},
                          "source_macro": e["source_macro_rmse"]})
    sdf = pd.DataFrame(seed_rows)
    R["seed_stability"] = {c: {"mean": float(sdf[c].mean()), "std": float(sdf[c].std())}
                           for c in ["NASA", "BatICM", "SNL", "source_macro"]}
    for c, v in R["seed_stability"].items():
        log(f"   {c:13s} RMSE mean {v['mean']:.4f} +- {v['std']:.4f}")

    # 7) 엔티티 단위 부트스트랩 신뢰구간 (test) ----------------------------------------------
    log("\n== [7] cluster bootstrap CI (test, resample entities)")
    B = 50 if args.quick else 1000
    rs = np.random.RandomState(SEED)
    em_m = entity_mse(test, p_test)
    em_b = entity_mse(test, SourceMedian().fit(dev).predict(test))
    ent_src = test.groupby("entity_id")["source"].first()
    ci = {}
    for s in sorted(test.source.unique()):
        ids = ent_src[ent_src == s].index.to_numpy()
        mm, bb = em_m[ids].to_numpy(), em_b[ids].to_numpy()
        rm, rb = [], []
        for _ in range(B):
            k = rs.randint(0, len(ids), len(ids))
            rm.append(np.sqrt(mm[k].mean())); rb.append(np.sqrt(bb[k].mean()))
        rm, rb = np.array(rm), np.array(rb)
        ci[s] = {"xgb_rmse_ci95": [float(np.percentile(rm, 2.5)), float(np.percentile(rm, 97.5))],
                 "baseline_rmse_ci95": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))],
                 "prob_xgb_better_than_baseline": float((rm < rb).mean()), "n_entities": int(len(ids))}
        log(f"   {s:7s} n={len(ids)} XGB {ci[s]['xgb_rmse_ci95'][0]:.4f}~{ci[s]['xgb_rmse_ci95'][1]:.4f} | "
            f"baseline {ci[s]['baseline_rmse_ci95'][0]:.4f}~{ci[s]['baseline_rmse_ci95'][1]:.4f} | "
            f"P(better)={ci[s]['prob_xgb_better_than_baseline']:.2f}")
    R["bootstrap_ci"] = ci

    # 8) 셔플 정답 sanity -- 누수가 있으면 정답을 섞어도 점수가 좋아진다 -----------------------------
    log("\n== [8] shuffled-target sanity check (expect: NOT better than source-median baseline)")
    shuf = dev.copy()
    shuf[TARGET] = shuf.groupby("source")[TARGET].transform(lambda s: rs.permutation(s.to_numpy()))
    bst_s = xgb.train(best_params, dmat(shuf, make_weights(shuf, best_mode)),
                      num_boost_round=n_rounds_dev, verbose_eval=False)
    sc_s = evaluate(test, predict(bst_s, test))["source_macro_rmse"]
    sc_b = base["source_median"]["test"]["source_macro_rmse"]
    R["shuffle_check"] = {"shuffled_model_test_rmse": sc_s, "baseline_rmse": sc_b, "passed": bool(sc_s >= sc_b * 0.98)}
    log(f"   shuffled-model {sc_s:.4f} vs baseline {sc_b:.4f} -> {'PASS' if R['shuffle_check']['passed'] else 'FAIL (LEAK?)'}")

    # 9) LOSO: 한 소스를 통째로 빼고 학습 -> 그 소스 예측 (소스 식별 피처 제외) -----------------------
    log("\n== [9] leave-one-source-out (informational)")
    feats_loso = [f for f in FEATURES if f not in SOURCE_ID_FEATURES]
    loso = {}
    for hold in ["NASA", "BatICM", "SNL"]:
        trn, hld = df[df.source != hold].reset_index(drop=True), df[df.source == hold].reset_index(drop=True)
        b = xgb.train(best_params, dmat(trn, make_weights(trn, best_mode), feats_loso),
                      num_boost_round=n_rounds_dev, verbose_eval=False)
        ph = predict(b, hld, feats_loso)
        e = evaluate(hld, ph)["per_source"][hold]
        mean_base = float(np.sqrt(np.mean((hld[TARGET] - trn[TARGET].mean()) ** 2)))
        loso[hold] = {"entity_macro_rmse": e["entity_macro_rmse"], "mae": e["mae"],
                      "r2_within_source": e["r2_within_source"], "baseline_train_mean_rmse": mean_base}
        log(f"   hold-out {hold:7s} RMSE {e['entity_macro_rmse']:.4f} | MAE {e['mae']:.4f} | "
            f"R2 {e['r2_within_source']:.3f} | (train-mean baseline {mean_base:.4f})")
    R["loso"] = loso

    # 10) 중요도 + 그림 ---------------------------------------------------------------------
    log("\n== [10] importance & plots")
    gain = bst.get_score(importance_type="gain")
    gain = {k: float(v) for k, v in sorted(gain.items(), key=lambda kv: -kv[1])}
    base_sc = R["xgb_test"]["source_macro_rmse"]
    perm = {}
    for f in FEATURES:
        inc = []
        for r in range(5):
            tp = test.copy()
            tp[f] = rs.permutation(tp[f].to_numpy())
            inc.append(evaluate(tp, predict(bst, tp))["source_macro_rmse"] - base_sc)
        perm[f] = float(np.mean(inc))
    perm = dict(sorted(perm.items(), key=lambda kv: -kv[1]))
    R["importance_gain"], R["importance_permutation_test"] = gain, perm
    log("   permutation importance (RMSE increase):")
    for k, v in list(perm.items())[:8]:
        log(f"     {k:20s} {v:+.5f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, s in zip(ax, ["NASA", "BatICM", "SNL"]):
        m = test.source == s
        a.scatter(test.loc[m, TARGET], p_test[m.to_numpy()], s=4, alpha=0.4)
        lo, hi = test.loc[m, TARGET].min(), test.loc[m, TARGET].max()
        a.plot([lo, hi], [lo, hi], "r--", lw=1)
        a.set_title(f"{s} (test)  RMSE={R['xgb_test']['per_source'][s]['entity_macro_rmse']:.4f}")
        a.set_xlabel("true SOH"); a.set_ylabel("pred SOH")
    plt.tight_layout(); plt.savefig(OUT / "pred_vs_true.png", dpi=130); plt.close()

    ents = []
    for s in ["NASA", "BatICM", "SNL"]:
        ents += list(test[test.source == s].entity_id.unique()[:3])
    fig, ax = plt.subplots(3, 3, figsize=(15, 10))
    for a, e in zip(ax.ravel(), ents):
        m = (test.entity_id == e).to_numpy()
        o = np.argsort(test.loc[m, "cycle_index"].to_numpy())
        a.plot(test.loc[m, "cycle_index"].to_numpy()[o], test.loc[m, TARGET].to_numpy()[o], ".", ms=2, label="true")
        a.plot(test.loc[m, "cycle_index"].to_numpy()[o], p_test[m][o], "-", lw=1, label="pred")
        a.set_title(e[:38], fontsize=8)
    ax[0, 0].legend()
    plt.tight_layout(); plt.savefig(OUT / "test_trajectories.png", dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    ks = list(perm)[::-1]
    ax.barh(ks, [perm[k] for k in ks]); ax.set_title("Permutation importance (test, source-macro RMSE increase)")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=130); plt.close()

    # 11) 저장: 평가용 모델(dev) + 배포용 모델(dev+test 전체 재학습) ---------------------------------
    bst.save_model(str(OUT / "xgb_soh_dev.json"))
    full = df.reset_index(drop=True)
    n_full = int(np.ceil(n_rounds_dev * len(full) / len(dev)))
    bst_full = xgb.train(best_params, dmat(full, make_weights(full, best_mode)),
                         num_boost_round=n_full, verbose_eval=False)
    bst_full.save_model(str(OUT / "xgb_soh_full.json"))
    meta = {"features": FEATURES, "target": TARGET, "weight_mode": best_mode,
            "params": R["best_params"], "n_rounds_dev": n_rounds_dev, "n_rounds_full": n_full,
            "note": "xgb_soh_dev.json = 평가에 쓴 모델(test 미포함). xgb_soh_full.json = 모든 데이터로 재학습한 배포용(성능 보고에 쓰지 말 것)."}
    (OUT / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    R["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(R)
    log(f"\n== done in {R['elapsed_sec']}s -> {OUT}")


def write_report(R):
    L = ["# SOH XGBoost 학습 리포트", "",
         f"- 환경: xgboost {R['env']['xgboost']} / sklearn {R['env']['sklearn']} / seed {R['env']['seed']}",
         f"- 데이터: dev {R['data']['dev_rows']}행·{R['data']['dev_entities']}개체 / test {R['data']['test_rows']}행·{R['data']['test_entities']}개체",
         f"- 정제: {json.dumps(R['cleaning'], ensure_ascii=False)}",
         f"- 선택된 가중치 방식: **{R['chosen_weight_mode']}**, 라운드 {R['n_rounds_dev']}", "",
         "## 소스별 test RMSE (엔티티 균등) — 베이스라인 대비", "",
         "| 소스 | XGBoost | 소스중앙값 베이스라인 | cycle선형 베이스라인 | 개선율 | MAE | R²(소스 내) |", "|---|---|---|---|---|---|---|"]
    for s, v in R["xgb_test_vs_baseline"].items():
        t = R["xgb_test"]["per_source"][s]
        L.append(f"| {s} | {v['xgb']:.4f} | {v['median_baseline']:.4f} | {v['linear_baseline']:.4f} | "
                 f"{v['improvement_vs_median']*100:+.1f}% | {t['mae']:.4f} | {t['r2_within_source']:.3f} |")
    L += ["", "## 부트스트랩 95% CI (test 엔티티 재표집)", "", "| 소스 | n | XGB RMSE | 베이스라인 RMSE | P(XGB가 더 좋음) |", "|---|---|---|---|---|"]
    for s, c in R["bootstrap_ci"].items():
        L.append(f"| {s} | {c['n_entities']} | {c['xgb_rmse_ci95'][0]:.4f}~{c['xgb_rmse_ci95'][1]:.4f} | "
                 f"{c['baseline_rmse_ci95'][0]:.4f}~{c['baseline_rmse_ci95'][1]:.4f} | {c['prob_xgb_better_than_baseline']:.2f} |")
    L += ["", "## 검증 체크", "",
          f"- 셔플 정답 검사: {'PASS' if R['shuffle_check']['passed'] else 'FAIL'} "
          f"(셔플 모델 {R['shuffle_check']['shuffled_model_test_rmse']:.4f} vs 베이스라인 {R['shuffle_check']['baseline_rmse']:.4f})",
          f"- SNL test 조건이 dev에도 존재(복제 셀): {R['sanity']['snl_test_conditions_also_in_dev']} → SNL test 점수는 낙관적",
          "", "## 소스 간 일반화 (LOSO, 참고용)", "", "| 제외한 소스 | RMSE | MAE | R² | train평균 베이스라인 RMSE |", "|---|---|---|---|---|"]
    for s, v in R["loso"].items():
        L.append(f"| {s} | {v['entity_macro_rmse']:.4f} | {v['mae']:.4f} | {v['r2_within_source']:.3f} | {v['baseline_train_mean_rmse']:.4f} |")
    L += ["", "## 모델 비교 (그룹 CV source-macro RMSE / test source-macro RMSE)", "",
          "| 모델 | CV | test |", "|---|---|---|",
          f"| 소스중앙값 | {R['baselines']['source_median']['cv']['source_macro_rmse']:.4f} | {R['baselines']['source_median']['test']['source_macro_rmse']:.4f} |",
          f"| cycle 선형 | {R['baselines']['source_linear_cycle']['cv']['source_macro_rmse']:.4f} | {R['baselines']['source_linear_cycle']['test']['source_macro_rmse']:.4f} |",
          f"| RandomForest | {R['random_forest']['cv']['source_macro_rmse']:.4f} | {R['random_forest']['test']['source_macro_rmse']:.4f} |",
          f"| XGBoost(튜닝) | {R['xgb_cv_best']['source_macro_rmse']:.4f} | {R['xgb_test']['source_macro_rmse']:.4f} |",
          "", f"선택 기준 rel-score(소스별 RMSE/중앙값 베이스라인, 낮을수록 좋음): XGB 기본 {R['search']['default_params_rel_score']:.4f} → 튜닝 {R['search']['best_cv_rel_score']:.4f}, RF {R['random_forest']['cv_rel_score']:.4f}",
          "", "## 소스 단독 학습 vs 통합 학습 (CV, 엔티티 균등 RMSE)", "", "| 소스 | 단독 | 통합 |", "|---|---|---|"]
    for s, v in R["ablation_source_only_vs_all"].items():
        L.append(f"| {s} | {v['only_this_source_rmse']:.4f} | {v['all_sources_rmse']:.4f} |")
    L += ["", "## 시드 안정성 (test RMSE, 5 seeds)", ""]
    for c, v in R["seed_stability"].items():
        L.append(f"- {c}: {v['mean']:.4f} ± {v['std']:.4f}")
    L += ["", "## 가중치 방식 비교 (기본 파라미터, CV rel-score)", "", "| 방식 | rel-score |", "|---|---|"]
    for m, v in R["weight_mode_comparison"].items():
        L.append(f"| {m} | {v['rel_score']:.4f} |")
    L += ["", "## 최적 하이퍼파라미터", "", "```json", json.dumps(R["best_params"], indent=2), "```"]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
