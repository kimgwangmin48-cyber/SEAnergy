"""
RandomForest vs XGBoost 비교 -- train_final.py 중첩 CV와 '같은 바깥 5-fold'(76개 엔티티 전부)에서 RF의 처음 보는 배터리 성능을 잰다.
RF는 튜닝하지 않은 기본 설정(300그루, min_samples_leaf=5)이고, 피처/가중치는 최종 모델과 같다 (RF는 NaN을 못 다뤄 -999로 채움).
결과: results/rf_comparison.json  (make_report.py 가 있으면 리포트에 포함)
"""
import json
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import soh_common as common
import train_final as F


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    meta = json.loads((F.RES / "model_meta.json").read_text(encoding="utf-8"))
    feats, mode = meta["features"], meta["design"]["weight_mode"]
    raw, _ = F.load_all()
    df, _ = common.clean(raw)
    df = df.reset_index(drop=True)
    oof = np.full(len(df), np.nan)
    for k, (tr, te) in enumerate(F.make_folds(df, 5, F.SEED + 77)):
        a = df.iloc[tr]
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.7, n_jobs=-1, random_state=F.SEED)
        rf.fit(a[feats].fillna(-999), a[F.TARGET], sample_weight=common.make_weights(a, mode))
        oof[te] = rf.predict(df.iloc[te][feats].fillna(-999))
        print(f"fold {k} done", flush=True)
    out = {}
    for s in F.SRC:
        mk = (df.source == s).to_numpy()
        out[s] = {**F.row_metrics(df[F.TARGET].to_numpy()[mk], oof[mk]), "CI95": F.cluster_ci(df[mk], oof[mk])}
        print(s, {k: round(v, 4) for k, v in out[s].items() if k != "CI95"})
    (F.RES / "rf_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
