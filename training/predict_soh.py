"""
학습된 SEAnergy SOH 모델로 예측하는 최소 예제 (5개 시드 XGBoost 평균).

    from predict_soh import load_model, predict_soh
    model = load_model()
    out = predict_soh(df, dataset="BatICM", model=model)   # soh_pred, soh_lo90, soh_hi90

df 는 model_meta.json 의 features 컬럼을 모두 가져야 한다 (없는 측정값은 NaN).
BatICM 세션 피처(v_at_50, dVdSOC_at_30_to_60, spread_mean, *_r20 ...)는 build_baticm_features.py 와 같은 방식으로 계산한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

RES = Path(__file__).resolve().parent / "results"
DATASET_RATED_AH = {"NASA": 2.0, "BatICM": 145.0, "SNL": 3.0}


def load_model(res_dir=RES):
    meta = json.loads((Path(res_dir) / "model_meta.json").read_text(encoding="utf-8"))
    boosters = []
    for k in range(meta["n_models"]):
        b = xgb.Booster()
        b.load_model(str(Path(res_dir) / f"xgb_soh_final_seed{k}.json"))
        boosters.append(b)
    return boosters, meta


def add_dataset_columns(df, dataset):
    """데이터셋 식별 플래그와 정격 용량 컬럼을 채운다 (학습 때와 같은 인코딩)."""
    df = df.copy()
    df["is_nasa"], df["is_baticm"], df["is_snl"] = int(dataset == "NASA"), int(dataset == "BatICM"), int(dataset == "SNL")
    df["rated_capacity_ah"] = DATASET_RATED_AH[dataset]
    return df


def predict_soh(df, dataset=None, model=None):
    boosters, meta = model if model is not None else load_model()
    if dataset is not None:
        df = add_dataset_columns(df, dataset)
    feats = meta["features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"입력에 없는 피처: {missing}")
    d = xgb.DMatrix(df[feats].to_numpy(float), feature_names=list(feats), missing=np.nan)
    p = np.mean([b.predict(d) for b in boosters], axis=0)
    out = pd.DataFrame({"soh_pred": p}, index=df.index)
    if dataset is not None:
        h = meta["interval_half_width_90"][dataset]
        out["soh_lo90"], out["soh_hi90"] = p - h, p + h
    return out
