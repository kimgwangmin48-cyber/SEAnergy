"""
SEAnergy SOH XGBoost -- 최종 파이프라인 (열화 피처 + 그룹 검증 + 중첩 CV + 5-시드 평균 모델)

단계
  0. 데이터: 팀 통합본 + BatICM 세션 피처(build_baticm_features.py) 결합, 무결성 assert
  1. 설계 선택(dev만): 피처 세트 / 단조 제약 / 손실함수(MSE vs MAE) / BatICM 가중치를 그룹 CV로 비교
     (최소 rel-score의 1% 이내면 더 단순한 설정을 고른다)
  2. 튜닝(dev만): 랜덤서치 + 그룹 CV
  3. test 1회 평가(11개 엔티티, 한 번도 안 쓴 데이터): 데이터셋별 MAE/RMSE/R2 + 부트스트랩 CI + 베이스라인/RF/v1 비교 + 셔플 검사
  4. 중첩 CV(전체 데이터): 바깥 5-fold / 안쪽 3-fold 튜닝 -> 모든 행에 대한 '처음 보는 배터리' 예측 (보고용 주 지표)
  5. 최종 모델: 전체 데이터로 5개 시드 재학습(평균) -> 모델 파일 + 행별 SOH 예측(+90% 구간)

검증 원칙: 같은 배터리/차량/SNL 시험조건은 항상 같은 fold. 설계/튜닝은 test를 보지 않는다. 보고 수치는 튜닝에 쓰이지 않은 데이터에서만.

속도/재현성
  - XGBoost 스레드 수는 4로 고정(soh_common.NTHREAD) -> 컴퓨터가 달라도 같은 결과.
  - fold(와 반복)를 스레드 풀로 동시에 학습하고, fold별 DMatrix는 한 번만 만들어 모든 후보가 재사용한다.
    병렬 정도(SOH_N_JOBS)는 결과에 영향이 없다(각 학습이 독립이고 스레드 수가 고정).

실행:  python build_baticm_features.py  ->  python train_final.py [--trials 12 --inner-trials 5] [--quick]
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
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

import soh_common as common
from soh_common import SEED, TARGET

warnings.filterwarnings("ignore")
SRC = ["NASA", "BatICM", "SNL"]
FE_PATH = common.OUT / "features_baticm.pkl"
RES = common.ROOT / "training" / "results"
BAG = 5  # 최종/평가 모델은 시드 5개 평균 (subsample/colsample 무작위성에 의한 분산 감소)
N_JOBS = int(os.environ.get("SOH_N_JOBS", max(1, min(5, (os.cpu_count() or 4) // common.NTHREAD))))

# v1(팀 통합본 16피처, MSE 손실, source_balanced) 최종 설정 -- 비교 기준선으로 재현
V1_PARAMS = dict(learning_rate=0.03742472165665569, max_depth=7, min_child_weight=47.735964446695256,
                 subsample=0.8550229885420852, colsample_bytree=0.9436063712881633, reg_lambda=1.8814553601769874,
                 reg_alpha=0.002769335216182953, gamma=0.01)
V1_ROUNDS = 201
# 탐색 시작점 3개: v1 설정 / '깊은 트리 + 강한 규제' / RandomForest처럼 분산을 줄이는 영역(깊은 트리 + 서브샘플링 + 노드별 컬럼 샘플링)
START_POINTS = [V1_PARAMS,
                dict(learning_rate=0.03, max_depth=8, min_child_weight=100, reg_lambda=10, subsample=0.8,
                     colsample_bytree=0.8, reg_alpha=0.0, gamma=0.0),
                dict(learning_rate=0.05, max_depth=12, min_child_weight=5, subsample=0.7, colsample_bytree=1.0,
                     colsample_bynode=0.5, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)]
NOISE_WINDOW = 101
# 정답(용량/SOH)을 복원하거나 BMS가 이미 계산해 둔 값 -> 입력 금지 (정격용량 rated_capacity_ah 는 소스별 상수라 허용)
LEAKY = {"capacity_ah", "capacity_ah_unified", "available_capacity", "available_energy", "duration_s", "n_samples"}


def log(m=""):
    print(m, flush=True)


# ------------------------------------------------------------------ 데이터 / 모델 입출력
def load_all():
    raw = common.load_data()
    fe = pd.read_pickle(FE_PATH)
    fe.columns = [c.replace("@", "_at_").replace("-", "_to_") for c in fe.columns]
    raw = raw.merge(fe, on="row_id", how="left")
    fe_cols = [c for c in fe.columns if c != "row_id"]
    assert raw.loc[raw.source == "BatICM", "t_mean"].notna().all(), "BatICM 피처 결합 실패"
    assert raw.loc[raw.source != "BatICM", fe_cols].isna().all().all(), "비-BatICM 행에 피처가 있음"
    return raw, fe_cols


def objective_params(obj):
    p = dict(common.BASE_PARAMS)
    if obj == "abs":
        p.update(objective="reg:absoluteerror", eval_metric="mae")
    return p


def mono_str(feats, decreasing=("cycle_index", "time_elapsed_days")):
    """SOH는 노화(사이클/경과일)에 대해 감소 방향으로 단조 -- 물리적으로 말이 되는 제약"""
    return "(" + ",".join("-1" if f in decreasing else "0" for f in feats) + ")"


def dm(df, feats, w=None):
    return xgb.DMatrix(df[feats].to_numpy(float), label=df[TARGET].to_numpy(float), weight=w,
                       feature_names=list(feats), missing=np.nan)


def pred(bst, df, feats):
    return bst.predict(xgb.DMatrix(df[feats].to_numpy(float), feature_names=list(feats), missing=np.nan))


def pmap(fn, items, workers=None):
    """스레드 풀 map (XGBoost 학습은 GIL을 놓으므로 스레드로 충분하고 데이터 복사가 없다)"""
    with ThreadPoolExecutor(workers or N_JOBS) as ex:
        return list(ex.map(fn, items))


# ------------------------------------------------------------------ 병렬 CV (DMatrix 캐시)
class CVData:
    """fold별 (검증 인덱스, 학습 DMatrix(가중치), 검증 DMatrix(엔티티 균등 가중))를 한 번만 만들어 후보마다 재사용"""

    def __init__(self, df, folds, mode, feats):
        def build(f):
            tr, va = f
            a, b = df.iloc[tr], df.iloc[va]
            return va, dm(a, feats, common.make_weights(a, mode)), dm(b, feats, common.make_weights(b, "entity"))
        self.n, self.folds = len(df), pmap(build, folds)


def _fit_fold(params, dtr, dva):
    bst = xgb.train(params, dtr, num_boost_round=4000, evals=[(dva, "valid")],
                    early_stopping_rounds=100, verbose_eval=False)
    n = int(bst.best_iteration) + 1
    return bst.predict(dva, iteration_range=(0, n)), n


def cv_fit(datas, params, feats, mono=False):
    """CVData 목록(반복)의 모든 fold를 동시에 학습. 조기종료는 fold 검증셋에서. -> (반복별 OOF 목록, 라운드 목록)"""
    p = dict(params)
    if mono:
        p["monotone_constraints"] = mono_str(feats)
    tasks = [(r, va, dtr, dva) for r, d in enumerate(datas) for va, dtr, dva in d.folds]
    results = pmap(lambda t: _fit_fold(p, t[2], t[3]), tasks)
    oofs, iters = [np.full(d.n, np.nan) for d in datas], []
    for (r, va, _, _), (pr, n) in zip(tasks, results):
        oofs[r][va] = pr
        iters.append(n)
    assert all(np.isfinite(o).all() for o in oofs), "OOF 결측"
    return oofs, iters


def fit_bag(params, df, feats, w, rounds, n_bag=BAG):
    def one(k):
        return xgb.train(dict(params, seed=SEED + k), dm(df, feats, w), num_boost_round=rounds, verbose_eval=False)
    return pmap(one, range(n_bag), min(N_JOBS, n_bag))


def pred_bag(bag, df, feats):
    d = xgb.DMatrix(df[feats].to_numpy(float), feature_names=list(feats), missing=np.nan)
    return np.mean([b.predict(d) for b in bag], axis=0)


def make_folds(df, n_splits, seed):
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(df, df["source"], groups=df["group"]))
    for tr, va in folds:
        assert set(df.iloc[tr].group).isdisjoint(set(df.iloc[va].group)), "GROUP LEAK"
    return folds


def baseline_oofs(df, fold_sets):
    out = []
    for fs in fold_sets:
        o = np.zeros(len(df))
        for tr, va in fs:
            o[va] = common.SourceMedian().fit(df.iloc[tr]).predict(df.iloc[va])
        out.append(o)
    return out


# ------------------------------------------------------------------ 지표
def row_metrics(y, p):
    e = y - p
    sst = ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "R2": float(1 - (e ** 2).sum() / sst) if sst > 0 else float("nan")}


def cluster_ci(d, p, B=1000):
    """그룹(엔티티; SNL은 시험조건) 단위 부트스트랩 95% CI"""
    y = d[TARGET].to_numpy(float)
    t = pd.DataFrame({"g": d["group"].to_numpy(), "n": 1.0, "ab": np.abs(y - p), "se": (y - p) ** 2,
                      "sy": y, "syy": y ** 2}).groupby("g")[["n", "ab", "se", "sy", "syy"]].sum().to_numpy()
    rs = np.random.RandomState(SEED)
    res = {"MAE": [], "RMSE": [], "R2": []}
    for _ in range(B):
        n, ab, se, sy, syy = t[rs.randint(0, len(t), len(t))].sum(0)
        sst = syy - sy ** 2 / n
        res["MAE"].append(ab / n); res["RMSE"].append(np.sqrt(se / n)); res["R2"].append(1 - se / sst if sst > 0 else np.nan)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in res.items()}


def per_source_metrics(df, p):
    return {s: row_metrics(df[TARGET].to_numpy()[(df.source == s).to_numpy()], p[(df.source == s).to_numpy()]) for s in SRC}


# ------------------------------------------------------------------ 랜덤서치
def sample_params(rng):
    """랜덤서치 공간 (분산을 줄이는 깊은 트리 + 서브샘플링 영역까지 포함)"""
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return dict(learning_rate=lu(0.02, 0.15), max_depth=int(rng.randint(4, 13)), min_child_weight=lu(1, 100),
                subsample=float(rng.uniform(0.5, 1.0)), colsample_bytree=float(rng.uniform(0.6, 1.0)),
                colsample_bynode=float(rng.uniform(0.3, 1.0)), reg_lambda=lu(0.1, 30), reg_alpha=lu(1e-3, 2),
                gamma=float(rng.choice([0.0, 0.01, 0.1])))


def search(df, datas, bmed, feats, n_trials, base, mono, tag=""):
    """시작점 + 랜덤 후보를 그룹 CV(rel-score)로 비교. 반환: (최고 후보, 시도 기록)"""
    rng = np.random.RandomState(SEED)
    cand = [dict(base, **sp) for sp in START_POINTS]
    for _ in range(n_trials):
        p = dict(base); p.update(sample_params(rng)); cand.append(p)
    best, rows = None, []
    for t, p in enumerate(cand):
        oofs, its = cv_fit(datas, p, feats, mono)
        sc = common.rel_score(df, oofs, bmed)
        rows.append({"trial": t, "rel_score": sc, "median_iters": int(np.median(its))})
        if best is None or sc < best["score"]:
            best = {"score": sc, "params": p, "iters": its}
            if tag:
                log(f"   {tag} trial {t:02d} rel-score {sc:.4f}  <-- best")
    return best, pd.DataFrame(rows)


# ------------------------------------------------------------------ 단계
def stage_design(dev, dfs, bmed, team16, full, R):
    """[1] 설계 선택: 단순 -> 복잡 순서로 비교하고, 최소 rel-score의 1% 이내면 더 단순한 설정을 고른다"""
    log("\n== [1] design selection (dev, grouped CV x%d, fixed v1 hyper-params)" % len(dfs))
    sq, ab = objective_params("sq"), objective_params("abs")
    cfgs = [("B_team16 (v1)", team16, sq, "source_balanced", False),
            ("C_full", full, sq, "source_balanced", False),
            ("C_full+monotone", full, sq, "source_balanced", True),
            ("+MAE loss", full, ab, "source_balanced", True),
            ("+BatICM weight x2", full, ab, "sb:BatICM=2", True),
            ("+BatICM weight x3", full, ab, "sb:BatICM=3", True)]
    y = dev[TARGET].to_numpy()
    sel = {}
    for name, feats, base, mode, mono in cfgs:
        datas = [CVData(dev, fs, mode, feats) for fs in dfs]
        oofs, _ = cv_fit(datas, dict(base, **V1_PARAMS), feats, mono)
        del datas
        ps = common.evaluate(dev, oofs[0])["per_source"]
        sel[name] = {"rel": common.rel_score(dev, oofs, bmed), **{s: ps[s]["entity_macro_rmse"] for s in SRC},
                     **{s + "_MAE": float(np.abs(y - oofs[0])[(dev.source == s).to_numpy()].mean()) for s in SRC}}
        v = sel[name]
        log(f"   {name:20s} rel {v['rel']:.4f} | BatICM RMSE {v['BatICM']:.4f} MAE {v['BatICM_MAE']:.4f}"
            f" | NASA {v['NASA']:.4f} | SNL {v['SNL']:.4f}")
    R["design_selection"] = sel
    best_rel = min(v["rel"] for v in sel.values())
    chosen = next(c for c in cfgs if sel[c[0]]["rel"] <= best_rel * 1.01)
    R["chosen_design"] = {"name": chosen[0], "objective": chosen[2]["objective"], "weight_mode": chosen[3],
                          "monotone": chosen[4], "n_features": len(chosen[1])}
    log(f"   -> chosen: {chosen[0]}  (rel {sel[chosen[0]]['rel']:.4f}, best {best_rel:.4f}; 1% 이내면 단순한 쪽)")
    return chosen


def stage_tune(dev, dfs, bmed, design, n_trials, R):
    """[2] 튜닝(dev): 랜덤서치 + 그룹 CV. 반환: (최종 파라미터, 개발 데이터 라운드 수)"""
    name, feats, base, mode, mono = design
    log(f"\n== [2] tuning on dev ({n_trials} random trials + {len(START_POINTS)} start points, grouped CV x{len(dfs)})")
    datas = [CVData(dev, fs, mode, feats) for fs in dfs]
    best, trials = search(dev, datas, bmed, feats, n_trials, base, mono, tag="dev")
    trials.to_csv(RES / "search_trials.csv", index=False)
    params = dict(best["params"])
    if mono:
        params["monotone_constraints"] = mono_str(feats)
    n_dev = int(np.ceil(np.median(best["iters"]) * 1.1))
    R["best_params"] = {k: v for k, v in params.items() if k not in ("seed", "nthread")}
    R["n_rounds_dev"] = n_dev
    R["tuning"] = {"design_rel": R["design_selection"][name]["rel"], "tuned_rel": float(best["score"]), "n_trials": n_trials}
    log(f"   rel-score design {R['design_selection'][name]['rel']:.4f} -> tuned {best['score']:.4f} | rounds {n_dev}")
    return params, n_dev


def stage_test(dev, test, design, params, n_dev, team16, R):
    """[3] test 1회 평가(+ 비교 모델, 시드 변동, 정답 셔플 검사)"""
    name, feats, base, mode, mono = design
    log("\n== [3] test (11 untouched entities, evaluated once)")
    wdev = common.make_weights(dev, mode)
    bag = fit_bag(params, dev, feats, wdev, n_dev)
    p_test = pred_bag(bag, test, feats)
    v1 = xgb.train(dict(objective_params("sq"), **V1_PARAMS, seed=SEED), dm(dev, team16, common.make_weights(dev, "source_balanced")),
                   num_boost_round=V1_ROUNDS, verbose_eval=False)
    p_v1 = pred(v1, test, team16)
    base_med, base_lin = common.SourceMedian().fit(dev).predict(test), common.SourceLinear().fit(dev).predict(test)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7, n_jobs=-1, random_state=SEED)
    rf.fit(dev[feats].fillna(-999), dev[TARGET], sample_weight=wdev)
    p_rf = rf.predict(test[feats].fillna(-999))
    tr = {}
    for s in SRC:
        mk = (test.source == s).to_numpy()
        y = test[TARGET].to_numpy()[mk]
        tr[s] = {"final": {**row_metrics(y, p_test[mk]), "CI95": cluster_ci(test[mk], p_test[mk])},
                 "v1_team16": row_metrics(y, p_v1[mk]), "random_forest": row_metrics(y, p_rf[mk]),
                 "median_baseline": row_metrics(y, base_med[mk]), "linear_trend_baseline": row_metrics(y, base_lin[mk]),
                 "n_rows": int(mk.sum()), "n_groups": int(test[mk].group.nunique())}
        v = tr[s]
        log(f"   {s:7s} FINAL MAE {v['final']['MAE']:.4f} RMSE {v['final']['RMSE']:.4f} R2 {v['final']['R2']:.3f} | v1 RMSE "
            f"{v['v1_team16']['RMSE']:.4f} | RF {v['random_forest']['RMSE']:.4f} | median {v['median_baseline']['RMSE']:.4f} | linear {v['linear_trend_baseline']['RMSE']:.4f}")
    R["test"] = tr
    # 단일 모델 시드 변동: 5-시드 모델의 각 멤버가 곧 시드별 단일 모델이다 (재학습 불필요)
    sd = pd.DataFrame([{s: m["RMSE"] for s, m in per_source_metrics(test, pred(b, test, feats)).items()} for b in bag])
    R["single_model_seed_std_test_RMSE"] = {s: [float(sd[s].mean()), float(sd[s].std())] for s in SRC}
    shuf = dev.copy()
    shuf[TARGET] = shuf.groupby("source")[TARGET].transform(lambda s: np.random.RandomState(SEED).permutation(s.to_numpy()))
    bs = xgb.train(params, dm(shuf, feats, common.make_weights(shuf, mode)), num_boost_round=n_dev, verbose_eval=False)
    sh, bm_ = common.evaluate(test, pred(bs, test, feats))["source_macro_rmse"], common.evaluate(test, base_med)["source_macro_rmse"]
    R["shuffle_check"] = {"shuffled_model": float(sh), "median_baseline": float(bm_), "passed": bool(sh >= 0.98 * bm_)}
    log(f"   shuffled-target check: {sh:.4f} vs baseline {bm_:.4f} -> {'PASS' if sh >= 0.98 * bm_ else 'FAIL'}")


def stage_nested(df, design, n_inner, R):
    """[4] 중첩 CV(전체 데이터). 반환: 행별 OOF 예측, 90% 구간 반폭(데이터셋별)"""
    name, feats, base, mode, mono = design
    log(f"\n== [4] nested CV on all data (outer 5 / inner 3 x {n_inner + len(START_POINTS)} candidates, bag {BAG})")
    oof, fold_id, info = np.full(len(df), np.nan), np.full(len(df), -1), []
    for k, (tr_i, te_i) in enumerate(make_folds(df, 5, SEED + 77)):
        trd = df.iloc[tr_i].reset_index(drop=True)
        ifs = [make_folds(trd, 3, SEED + 100 + k)]
        datas = [CVData(trd, ifs[0], mode, feats)]
        bk, _ = search(trd, datas, baseline_oofs(trd, ifs), feats, n_inner, base, mono)
        pk = dict(bk["params"])
        if mono:
            pk["monotone_constraints"] = mono_str(feats)
        nk = int(np.ceil(np.median(bk["iters"]) * 1.1))
        oof[te_i] = pred_bag(fit_bag(pk, trd, feats, common.make_weights(trd, mode), nk), df.iloc[te_i], feats)
        fold_id[te_i] = k
        info.append({"fold": k, "rounds": nk, "inner_rel": bk["score"], "max_depth": pk["max_depth"],
                     "learning_rate": round(pk["learning_rate"], 4)})
        log(f"   outer fold {k}: inner rel {bk['score']:.4f} | rounds {nk} | depth {pk['max_depth']} lr {pk['learning_rate']:.3f}")
    assert np.isfinite(oof).all()
    R["nested_cv_folds"] = info
    df["soh_pred_oof"] = oof

    nested = {}
    for s in SRC:
        mk = (df.source == s).to_numpy()
        y, p_ = df[TARGET].to_numpy()[mk], oof[mk]
        nested[s] = {**row_metrics(y, p_), "CI95": cluster_ci(df[mk], p_),
                     "median_baseline": row_metrics(y, common.SourceMedian().fit(df).predict(df[mk])),
                     "n_rows": int(mk.sum()), "n_groups": int(df[mk].group.nunique())}
        v = nested[s]
        log(f"   {s:7s} MAE {v['MAE']:.4f} [{v['CI95']['MAE'][0]:.4f},{v['CI95']['MAE'][1]:.4f}] | RMSE {v['RMSE']:.4f} "
            f"[{v['CI95']['RMSE'][0]:.4f},{v['CI95']['RMSE'][1]:.4f}] | R2 {v['R2']:.3f} [{v['CI95']['R2'][0]:.3f},{v['CI95']['R2'][1]:.3f}]"
            f" (median-baseline RMSE {v['median_baseline']['RMSE']:.4f})")
    R["nested_cv"] = nested

    pe = pd.DataFrame([{"dataset": s, "entity_id": e, "n_rows": len(d), **row_metrics(d[TARGET].to_numpy(), d["soh_pred_oof"].to_numpy())}
                       for (s, e), d in df.groupby(["source", "entity_id"])])
    pe.to_csv(RES / "metrics_per_entity.csv", index=False, encoding="utf-8-sig")
    bm = pe[pe.dataset == "BatICM"]
    R["BatICM_per_vehicle_MAE"] = {"min": float(bm.MAE.min()), "median": float(bm.MAE.median()), "max": float(bm.MAE.max()),
                                   "worst": bm.sort_values("MAE").entity_id.iloc[-1]}
    log(f"   BatICM per-vehicle MAE: min {bm.MAE.min():.4f} / median {bm.MAE.median():.4f} / max {bm.MAE.max():.4f} ({R['BatICM_per_vehicle_MAE']['worst']})")

    ares = np.abs(df[TARGET].to_numpy() - oof)
    cov, qs = {}, {}
    for s in SRC:  # 예측구간: 해당 fold를 뺀 OOF 절대잔차 90% 분위수로 fold별 커버리지 검증
        mk = (df.source == s).to_numpy()
        ins = [ares[mk & (fold_id == k)] <= np.quantile(ares[mk & (fold_id != k)], 0.90) for k in range(5) if (mk & (fold_id == k)).any()]
        cov[s], qs[s] = float(np.concatenate(ins).mean()), float(np.quantile(ares[mk], 0.90))
    R["interval_90"] = {"coverage_cv": cov, "half_width": qs}
    log("   90% interval: " + ", ".join(f"{s} +-{qs[s]:.4f} (coverage {cov[s]:.2f})" for s in SRC))

    b = df[df.source == "BatICM"].copy()
    b["res"] = b[TARGET] - b["soh_pred_oof"]
    ea = {}
    for nm, col in [("cycle_index", b.cycle_index), ("temperature_c", b.temperature_c), ("delta_soc", b.soc_end_frac - b.soc_start_frac)]:
        q = pd.qcut(col, 5, labels=False, duplicates="drop")
        ea[nm] = [{"quintile": int(i), "MAE": float(g.res.abs().mean()), "bias": float(g.res.mean())} for i, g in b.groupby(q)]
    R["BatICM_error_by_quintile"] = ea
    return qs


def stage_final(raw, df, design, params, n_dev, dev_len, qs, R):
    """[5] 전체 데이터로 5-시드 최종 모델 학습 -> 모델/설정/행별 예측 저장. 반환: 최종 5개 모델, 중요도"""
    name, feats, base, mode, mono = design
    log("\n== [5] final model on all data (5-seed bag)")
    n_full = int(np.ceil(n_dev * len(df) / dev_len))
    bag = fit_bag(params, df, feats, common.make_weights(df, mode), n_full)
    for k, b_ in enumerate(bag):
        b_.save_model(str(RES / f"xgb_soh_final_seed{k}.json"))
    imp = pd.concat([pd.Series(b_.get_score(importance_type="gain")) for b_ in bag], axis=1).mean(axis=1).sort_values(ascending=False)
    R["importance_gain"] = {k: float(v) for k, v in imp.items()}
    out = raw.copy()
    out["soh_pred"] = pred_bag(bag, out, feats)
    out = out.merge(df[["row_id", "soh_pred_oof"]], on="row_id", how="left")
    q = out["source"].map(qs)
    pd.DataFrame({
        "row_id": out.row_id, "dataset": out.source, "entity_id": out.entity_id, "cycle_index": out.cycle_index,
        "soh_true": out[TARGET], "soh_pred": out.soh_pred, "soh_pred_oof": out.soh_pred_oof,
        "soh_lo90": out.soh_pred - q, "soh_hi90": out.soh_pred + q,
        "abs_err": (out[TARGET] - out.soh_pred).abs(), "abs_err_oof": (out[TARGET] - out.soh_pred_oof).abs(),
        "excluded_outlier": out.row_id.isin(set(raw.row_id) - set(df.row_id)).astype(int)
    }).to_csv(RES / "soh_predictions_all_rows.csv", index=False, encoding="utf-8-sig")
    R["in_sample_final_model"] = per_source_metrics(df, pred_bag(bag, df, feats))
    meta = {"features": feats, "target": TARGET, "n_models": BAG, "n_rounds": n_full, "params": R["best_params"],
            "design": R["chosen_design"], "interval_half_width_90": qs, "data_sha256": R["env"]["data_sha256"],
            "note": "예측 = 5개 시드 모델 평균. soh_lo90/hi90 = 예측 +- 해당 데이터셋 OOF 절대잔차 90% 분위수. "
                    "BatICM 세션 피처는 build_baticm_features.py 로 생성."}
    (RES / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return imp


def save_figures(df, nested, imp):
    oof = df["soh_pred_oof"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a_, s in zip(ax, SRC):
        mk = (df.source == s).to_numpy()
        a_.scatter(df[TARGET][mk], oof[mk], s=2, alpha=0.3)
        lo, hi = df[TARGET][mk].min(), df[TARGET][mk].max()
        a_.plot([lo, hi], [lo, hi], "r--", lw=1)
        a_.set_title(f"{s} nested-CV (unseen battery)  MAE {nested[s]['MAE']:.4f}  R2 {nested[s]['R2']:.3f}")
        a_.set_xlabel("true SOH"); a_.set_ylabel("pred SOH")
    plt.tight_layout(); plt.savefig(RES / "nested_cv_pred_vs_true.png", dpi=130); plt.close()
    b = df[df.source == "BatICM"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a_, e in zip(ax.ravel(), list(b.entity_id.unique()[:6])):
        d = b[b.entity_id == e].sort_values("cycle_index")
        a_.plot(d.cycle_index, d[TARGET], ".", ms=1.5, label="true")
        a_.plot(d.cycle_index, d.soh_pred_oof, "-", lw=0.8, label="pred (unseen vehicle)")
        a_.set_title(e)
    ax[0, 0].legend(); plt.tight_layout(); plt.savefig(RES / "baticm_trajectories_oof.png", dpi=110); plt.close()
    fig, ax = plt.subplots(figsize=(7, 8))
    ks = list(imp.index[:25])[::-1]
    ax.barh(ks, [imp[k] for k in ks]); ax.set_title("Feature importance (gain, 5-seed mean)")
    plt.tight_layout(); plt.savefig(RES / "feature_importance.png", dpi=120); plt.close()


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--inner-trials", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.trials, a.inner_trials = 2, 1
    sys.stdout.reconfigure(encoding="utf-8")
    RES.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = {"env": {"xgboost": xgb.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__,
                 "numpy": np.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
                 "seed": SEED, "bag": BAG, "nthread": common.NTHREAD,
                 "data_sha256": hashlib.sha256(common.DATA_XLSX.read_bytes()).hexdigest()[:16]}}

    # ---------- 0. 데이터
    raw, fe_cols = load_all()
    df, clean_rep = common.clean(raw)
    df = df.reset_index(drop=True)
    dev = df[df.split.isin(["train", "validation"])].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    R["sanity"] = common.sanity_checks(df, dev, test)
    team16 = list(common.FEATURES)
    full = team16 + fe_cols
    for f in full:  # 누수 위험 컬럼 차단
        assert f not in common.FORBIDDEN_IN_X and f not in LEAKY and not f.startswith("available_"), f
    R["cleaning"] = clean_rep
    R["features"] = {"team16": team16, "baticm_session_features": fe_cols}
    log(f"== data: {len(df)} rows | dev {len(dev)} ({dev.entity_id.nunique()} ent) | test {len(test)} ({test.entity_id.nunique()} ent) "
        f"| +{len(fe_cols)} BatICM features | threads {common.NTHREAD} x jobs {N_JOBS}")

    # BatICM 정답 잡음 바닥 (정답이 차량 추세 주위에서 흔들리는 정도 -- 이보다 잘 맞추려면 피처에 정보가 있어야 한다)
    bb = df[df.source == "BatICM"].sort_values(["entity_id", "cycle_index"])
    trend = bb.groupby("entity_id")[TARGET].transform(lambda s: s.rolling(NOISE_WINDOW, center=True, min_periods=NOISE_WINDOW // 2).median())
    e_ = (bb[TARGET] - trend).dropna()
    R["BatICM_label_noise_floor_vs_local_trend"] = {"MAE": float(e_.abs().mean()), "RMSE": float(np.sqrt((e_ ** 2).mean()))}
    log(f"   BatICM label noise vs local trend: MAE {e_.abs().mean():.4f} RMSE {np.sqrt((e_ ** 2).mean()):.4f}")

    n_rep = 1 if a.quick else 3
    dfs = [make_folds(dev, 5, SEED + r) for r in range(n_rep)]
    bmed = baseline_oofs(dev, dfs)

    design = stage_design(dev, dfs, bmed, team16, full, R)          # [1]
    params, n_dev = stage_tune(dev, dfs, bmed, design, a.trials, R)  # [2]
    stage_test(dev, test, design, params, n_dev, team16, R)           # [3]
    qs = stage_nested(df, design, a.inner_trials, R)                  # [4]
    imp = stage_final(raw, df, design, params, n_dev, len(dev), qs, R)  # [5]
    save_figures(df, R["nested_cv"], imp)

    R["elapsed_min"] = round((time.time() - t0) / 60, 1)
    (RES / "metrics.json").write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n== done in {R['elapsed_min']} min -> training/results")


if __name__ == "__main__":
    main()
