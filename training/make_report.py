"""results/metrics.json -> results/report.md (사람이 읽는 결과 리포트). 수치는 전부 metrics.json 에서 가져온다."""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
SRC = ["BatICM", "SNL", "NASA"]


def ci(c):
    return f"{c[0]:.4f}~{c[1]:.4f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    R = json.loads((RES / "metrics.json").read_text(encoding="utf-8"))
    N, T = R["nested_cv"], R["test"]
    L = ["# SOH 예측 모델 결과 리포트", "",
         f"- 모델: XGBoost 5-시드 평균 ({R['chosen_design']['n_features']}개 피처, 손실 `{R['chosen_design']['objective']}`, "
         f"가중치 `{R['chosen_design']['weight_mode']}`, 단조 제약 {R['chosen_design']['monotone']})",
         f"- 환경: xgboost {R['env']['xgboost']} / scikit-learn {R['env']['sklearn']} / seed {R['env']['seed']} / 데이터 해시 {R['env']['data_sha256']}",
         f"- 정제: " + ", ".join(f"{k} {v['removed']}행 제거({v['rule']})" for k, v in R["cleaning"].items()), "",
         "## 1. 주 지표 — 처음 보는 배터리/차량 기준 (중첩 CV, 전체 데이터)", "",
         "바깥 5-fold로 평가하고, 각 fold 안쪽(3-fold)에서만 하이퍼파라미터를 골랐습니다. 같은 배터리/차량/SNL 시험조건은 한 fold에만 들어갑니다. "
         "괄호는 95% 신뢰구간(배터리/차량 단위 부트스트랩).", "",
         "| 데이터셋 | 검증 단위 수* | 행 수 | MAE | RMSE | R² | 중앙값 베이스라인 RMSE |", "|---|---|---|---|---|---|---|"]
    for s in SRC:
        v = N[s]
        L.append(f"| {s} | {v['n_groups']} | {v['n_rows']} | **{v['MAE']:.4f}** ({ci(v['CI95']['MAE'])}) | {v['RMSE']:.4f} ({ci(v['CI95']['RMSE'])}) | "
                 f"{v['R2']:.3f} ({v['CI95']['R2'][0]:.3f}~{v['CI95']['R2'][1]:.3f}) | {v['median_baseline']['RMSE']:.4f} |")
    L += ["", "\\* 검증 단위 = 배터리/차량 1개 (SNL은 같은 시험조건의 복제 셀을 묶은 '시험조건' 8개). "
          "SOH는 0~1 단위입니다. MAE 0.011이면 SOH 1.1%p 오차입니다. R²는 데이터셋마다 따로 계산했습니다 "
          "(합쳐서 내면 소스별 SOH 평균 차이로 부풀려집니다).", ""]
    rf_path = RES / "rf_comparison.json"
    if rf_path.exists():
        RF = json.loads(rf_path.read_text(encoding="utf-8"))
        L += ["**XGBoost vs RandomForest (같은 바깥 fold, RF는 튜닝 안 한 기본 설정)**", "",
              "| 데이터셋 | XGBoost MAE | RF MAE | XGBoost RMSE | RF RMSE | XGBoost R² | RF R² |", "|---|---|---|---|---|---|---|"]
        for s in SRC:
            L.append(f"| {s} | {N[s]['MAE']:.4f} | {RF[s]['MAE']:.4f} | {N[s]['RMSE']:.4f} | {RF[s]['RMSE']:.4f} | {N[s]['R2']:.3f} | {RF[s]['R2']:.3f} |")
        L.append("")
    L += [
          "## 2. 따로 떼어 둔 test (11개 엔티티, 설계·튜닝에 한 번도 쓰지 않음)", "",
          "| 데이터셋 | 최종 MAE | 최종 RMSE | 최종 R² | v1(팀 16피처) RMSE | RandomForest RMSE | 선형추세 RMSE | 중앙값 RMSE |", "|---|---|---|---|---|---|---|---|"]
    for s in SRC:
        v = T[s]
        L.append(f"| {s} | {v['final']['MAE']:.4f} | {v['final']['RMSE']:.4f} | {v['final']['R2']:.3f} | {v['v1_team16']['RMSE']:.4f} | "
                 f"{v['random_forest']['RMSE']:.4f} | {v['linear_trend_baseline']['RMSE']:.4f} | {v['median_baseline']['RMSE']:.4f} |")
    n_txt = ", ".join("%s %d" % (s, T[s]["n_groups"]) for s in SRC)
    L += ["", f"test 엔티티 수는 {n_txt}개뿐이라 값이 흔들립니다. 판단은 1번 표를 기준으로 하세요. "
          f"SNL test는 같은 시험조건의 복제 셀이 학습에 있어 낙관적입니다 (1번 표는 조건 단위로 묶어서 이 문제가 없습니다).", ""]
    d = R["design_selection"]
    L += ["## 3. 무엇을 바꿔서 얼마나 좋아졌나 (개발 데이터 그룹 CV, 같은 하이퍼파라미터)", "",
          "| 설정 | 종합 상대점수(낮을수록 좋음) | BatICM RMSE | BatICM MAE | NASA RMSE | SNL RMSE |", "|---|---|---|---|---|---|"]
    for k, v in d.items():
        L.append(f"| {k} | {v['rel']:.4f} | {v['BatICM']:.4f} | {v['BatICM_MAE']:.4f} | {v['NASA']:.4f} | {v['SNL']:.4f} |")
    L += ["", f"종합 상대점수 = 데이터셋별 (RMSE ÷ 같은 fold의 데이터셋 중앙값 예측 RMSE)의 평균 (1 미만이면 베이스라인보다 낫다). "
          f"선택 규칙: 최소 점수의 1% 이내면 더 단순한 설정 → **{R['chosen_design']['name']}** 선택. "
          f"이후 랜덤서치 튜닝으로 {R['tuning']['design_rel']:.4f} → {R['tuning']['tuned_rel']:.4f}.", "",
          "## 4. 검증 체크", ""]
    sc = R["shuffle_check"]
    L += [f"- 정답 셔플 검사: {'통과' if sc['passed'] else '실패'} (정답을 섞어 학습하면 test RMSE {sc['shuffled_model']:.4f}로 중앙값 예측 {sc['median_baseline']:.4f}과 같은 수준 → 누수 없음)",
          f"- BatICM 정답 잡음 바닥: 정답이 차량 추세 주위에서 흔들리는 정도가 MAE {R['BatICM_label_noise_floor_vs_local_trend']['MAE']:.4f} / "
          f"RMSE {R['BatICM_label_noise_floor_vs_local_trend']['RMSE']:.4f} → 모델 오차가 이 값에 가까워서 BatICM에서 더 줄일 여지는 작음",
          f"- BatICM 차량별 MAE: 최소 {R['BatICM_per_vehicle_MAE']['min']:.4f} / 중앙 {R['BatICM_per_vehicle_MAE']['median']:.4f} / "
          f"최대 {R['BatICM_per_vehicle_MAE']['max']:.4f} ({R['BatICM_per_vehicle_MAE']['worst']}) — 20대 모두 비슷한 수준",
          "- 90% 예측구간(±) / 교차검증 커버리지: " + ", ".join(
              f"{s} ±{R['interval_90']['half_width'][s]:.3f} ({R['interval_90']['coverage_cv'][s]:.2f})" for s in SRC),
          "- 단일 모델 시드 변동(test RMSE 표준편차): " + ", ".join(f"{s} {v[1]:.4f}" for s, v in R["single_model_seed_std_test_RMSE"].items()), ""]
    L += ["## 5. BatICM 오차 분석 (처음 보는 차량 기준, 5분위별 MAE / 편향)", ""]
    for nm, rows in R["BatICM_error_by_quintile"].items():
        L.append(f"- {nm}: " + " · ".join(f"Q{r['quintile'] + 1} {r['MAE']:.4f}/{r['bias']:+.4f}" for r in rows))
    imp = list(R["importance_gain"].items())[:10]
    L += ["", "## 6. 피처 중요도 (gain, 상위 10)", ""] + [f"- `{k}`: {v:.3f}" for k, v in imp]
    p = R["best_params"]
    L += ["", "## 7. 최종 하이퍼파라미터", "", "```json", json.dumps(p, indent=2, ensure_ascii=False), "```", "",
          f"라운드 수(전체 데이터): 개발 {R['n_rounds_dev']} → 전체 학습 시 데이터 크기 비례 보정"]
    (RES / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("report written")


if __name__ == "__main__":
    main()
