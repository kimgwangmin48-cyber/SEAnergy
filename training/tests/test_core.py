"""
핵심 로직 빠른 테스트 (합성 데이터, 데이터셋 없이 수 초 안에 실행).

    python training/tests/test_core.py        # pytest가 있으면:  pytest training/tests

검사 항목: 그룹 fold 누수 없음 / 가중치 성질 / 병렬·순차·캐시 재사용이 같은 결과 / 정제 규칙 / 입력 컬럼 누수 가드 / 추론 래퍼 왕복
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import predict_soh  # noqa: E402
import soh_common as common  # noqa: E402
import train_final as F  # noqa: E402

FEATS = ["cycle_index", "temperature_c", "is_nasa", "is_baticm", "is_snl", "rated_capacity_ah"]


def synthetic(n_per=40, seed=0):
    rs = np.random.RandomState(seed)
    rows = []
    spec = {"NASA": (6, 2.0), "BatICM": (5, 145.0), "SNL": (6, 3.0)}
    for src, (n_ent, rated) in spec.items():
        for e in range(n_ent):
            c = np.arange(1, n_per + 1)
            soh = 1.0 - 0.003 * c * (1 + 0.3 * e / n_ent) + rs.normal(0, 0.004, n_per)
            ent = f"{src}_{e}" if src != "SNL" else f"SNL_cond{e // 2}_{'ab'[e % 2]}"
            rows.append(pd.DataFrame({"source": src, "entity_id": ent, "cycle_index": c, "temperature_c": 25 + e,
                                      "rated_capacity_ah": rated, common.TARGET: soh}))
    df = pd.concat(rows, ignore_index=True)
    for s, col in [("NASA", "is_nasa"), ("BatICM", "is_baticm"), ("SNL", "is_snl")]:
        df[col] = (df.source == s).astype(int)
    df["group"] = np.where(df.source == "SNL", df.entity_id.str.replace(r"_[a-z]$", "", regex=True), df.entity_id)
    return df


def test_group_folds_have_no_leak():
    df = synthetic()
    for tr, va in F.make_folds(df, 3, seed=1):
        assert set(df.iloc[tr].group).isdisjoint(set(df.iloc[va].group))
        assert len(tr) + len(va) == len(df)
    # SNL 복제 셀(a/b)은 같은 시험조건이라 항상 같은 fold
    snl = df[df.source == "SNL"]
    assert snl.groupby("group").entity_id.nunique().max() == 2


def test_weights_properties():
    df = synthetic()
    for mode in ["none", "entity", "source_balanced", "sb:BatICM=3"]:
        w = common.make_weights(df, mode)
        assert abs(w.mean() - 1) < 1e-9 and (w > 0).all(), mode
    w = common.make_weights(df, "entity")
    per_ent = pd.Series(w).groupby(df.entity_id.to_numpy()).sum()
    assert np.allclose(per_ent, per_ent.iloc[0])  # 배터리 1개 = 1표
    w = common.make_weights(df, "source_balanced")
    per_src = pd.Series(w).groupby(df.source.to_numpy()).sum()
    assert np.allclose(per_src, per_src.iloc[0])  # 소스 3개 동일 비중
    w3 = common.make_weights(df, "sb:BatICM=3")
    assert pd.Series(w3).groupby(df.source.to_numpy()).sum()["BatICM"] > per_src["BatICM"]


def test_cv_is_deterministic_across_parallelism_and_cache():
    df = synthetic()
    params = dict(F.objective_params("abs"), max_depth=3, learning_rate=0.1, nthread=2)
    folds = F.make_folds(df, 3, seed=2)
    datas = [F.CVData(df, folds, "source_balanced", FEATS)]
    old_jobs = F.N_JOBS
    try:
        F.N_JOBS = 3
        a, it_a = F.cv_fit(datas, params, FEATS, mono=True)
        F.cv_fit(datas, dict(params, max_depth=2), FEATS)  # 다른 후보로 캐시 재사용
        b, it_b = F.cv_fit(datas, params, FEATS, mono=True)
        F.N_JOBS = 1
        c, it_c = F.cv_fit([F.CVData(df, folds, "source_balanced", FEATS)], params, FEATS, mono=True)
    finally:
        F.N_JOBS = old_jobs
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
    assert sorted(it_a) == sorted(it_b) == sorted(it_c)


def test_clean_rules_and_sanity():
    df = synthetic()
    df.loc[df.index[0], common.TARGET] = 5.0  # 비정상 SOH
    out, rep = common.clean(df)
    assert len(out) == len(df) - 1 and rep["NASA"]["removed"] == 1
    assert not (set(common.FEATURES) & common.FORBIDDEN_IN_X)


def test_input_columns_exclude_leaky_fields():
    for f in common.FEATURES:
        assert f not in F.LEAKY and not f.startswith("available_") and f not in common.FORBIDDEN_IN_X
    meta_path = F.RES / "model_meta.json"
    if meta_path.exists():  # 학습된 모델이 있으면 그 입력 컬럼도 검사
        for f in json.loads(meta_path.read_text(encoding="utf-8"))["features"]:
            assert f not in F.LEAKY and not f.startswith("available_") and f not in common.FORBIDDEN_IN_X, f


def test_predict_wrapper_roundtrip():
    df = synthetic()
    params = dict(F.objective_params("sq"), max_depth=3, nthread=2)
    bag = F.fit_bag(params, df, FEATS, common.make_weights(df, "entity"), 20, n_bag=2)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for k, b in enumerate(bag):
            b.save_model(str(d / f"xgb_soh_final_seed{k}.json"))
        meta = {"features": FEATS, "n_models": 2, "interval_half_width_90": {"BatICM": 0.02}}
        (d / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        model = predict_soh.load_model(d)
        sub = df[df.source == "BatICM"].drop(columns=["is_nasa", "is_baticm", "is_snl", "rated_capacity_ah"])
        out = predict_soh.predict_soh(sub, dataset="BatICM", model=model)
        ref = F.pred_bag(bag, predict_soh.add_dataset_columns(sub, "BatICM"), FEATS)
        assert np.allclose(out.soh_pred.to_numpy(), ref, atol=1e-6)
        assert np.allclose(out.soh_hi90 - out.soh_pred, 0.02)
        try:
            predict_soh.predict_soh(sub.drop(columns=["cycle_index"]), dataset="BatICM", model=model)
            raise AssertionError("missing feature must raise")
        except ValueError:
            pass


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
