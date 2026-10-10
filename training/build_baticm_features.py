"""
BatICM 원본(8초 간격 CAN 로그)에서 '충전 세션별 열화 피처'를 만든다.

[왜 필요한가]
통합본의 BatICM 입력은 cycle_index / 경과일 / 온도 / SOC 시작·종료 5개뿐이라, 모델이 사실상 '시간 추세'만 배운다.
실제 열화 정보는 충전 중 전압 곡선(같은 SOC에서 전압이 얼마나 높은가 = 내부저항, SOC당 전압 기울기 = 용량 관련)과
셀 간 전압 편차, 온도에 들어 있다. 이를 세션 단위로 요약한다.

[누수 방지 원칙 -- 처음 보는 사람이 물을 만한 지점]
정답(SOH)은 "충전 전류 적분 / SOC 변화량"(쿨롱 카운팅)으로 만든 값이다. 따라서
  - 전류 x 시간으로 용량을 복원할 수 있는 피처는 쓰지 않는다: 세션 지속시간, 샘플 수, 시간 기준 기울기(dV/dt), SOC 변화 속도
  - BMS가 직접 계산해 둔 available_capacity / available_energy 컬럼은 쓰지 않는다 (SOH와 사실상 같은 정보)
  - 사용: SOC 기준 전압(V@SOC), SOC당 전압 기울기(dV/dSOC), 해당 SOC에서의 전류 크기, 셀 전압 편차, 온도
  - 롤링 집계는 '현재 + 과거 세션'만 쓰는 인과적(trailing) 윈도우 -> 미래 정보 사용 없음

세션 분할/채택 규칙은 extract_capacity.py(원 저자 코드 기반)와 동일하게 재현하고,
재계산한 용량이 processed 표와 일치하는지 assert로 검증한 뒤 통합본 행과 순서 정렬로 대응시킨다.

출력: training/outputs/features_baticm.pkl  (통합본 BatICM 행과 1:1, row_id 키)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]  # 저장소 루트 (training/ 의 상위 폴더)
RAW_DIR = ROOT / "datasets" / "BatICM" / "raw_csv"
PROCESSED = ROOT / "datasets" / "BatICM" / "processed" / "batICM_capacity_per_cycle.csv"
MERGED_CACHE = ROOT / "training" / "outputs" / "_data_cache.pkl"
OUT = ROOT / "training" / "outputs" / "features_baticm.pkl"

COLS = ["number", "record_time", "soc", "pack_voltage", "charge_current", "max_cell_voltage",
        "min_cell_voltage", "max_temperature", "min_temperature", "available_energy", "available_capacity"]
SOC_POINTS = [30, 50, 70, 90]
SOC_WINDOWS = [(30, 60), (60, 90)]
HALF_BIN = 1.0           # SOC +-1% 안의 샘플 중앙값 사용 (전류 변동 노이즈 완화)
MIN_SAMPLES_BIN = 3
ROLL = 20                # 인과적 롤링 윈도우(세션 수)


def find_segments(df):
    """extract_capacity.find_samples_in_file 과 동일한 분할 규칙 (마지막 행 제외 규칙 포함)"""
    df = df.reset_index(drop=True)
    t = pd.to_datetime(df["record_time"].astype(str))
    gap = np.where((t.diff().shift(-1) > pd.Timedelta(seconds=10)).to_numpy())[0]
    if len(gap) == 0:
        return [df]
    segs = [df.iloc[: gap[0]]]
    for i in range(len(gap) - 1):
        segs.append(df.iloc[gap[i] + 1: gap[i + 1]])
    segs.append(df.iloc[gap[-1] + 1:])
    return segs


def capacity_ah(seg):
    """원 저자 방식 쿨롱 카운팅 (정답 재현 검증용, 피처로는 쓰지 않음)"""
    cur = seg["charge_current"].to_numpy(float).copy()
    if np.isnan(cur).sum() > len(cur) * 0.1:
        return 0.0
    if np.isnan(cur).any():
        for n in range(len(cur)):
            if np.isnan(cur[n]):
                cur[n] = cur[n - 1]
    t = pd.to_datetime(seg["record_time"].astype(str))
    sec = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    q = np.trapezoid(cur, sec) / 3600 * (-1)
    dsoc = seg["soc"].iloc[-1] - seg["soc"].iloc[0]
    return 0.0 if dsoc == 0 else q / dsoc * 100


def session_features(seg):
    soc = seg["soc"].to_numpy(float)
    v = seg["pack_voltage"].to_numpy(float)
    i = np.abs(seg["charge_current"].to_numpy(float))
    spread = (seg["max_cell_voltage"] - seg["min_cell_voltage"]).to_numpy(float)
    tmax, tmin = seg["max_temperature"].to_numpy(float), seg["min_temperature"].to_numpy(float)
    tmean = (tmax + tmin) / 2
    f = {
        "t_mean": np.nanmean(tmean), "t_max": np.nanmean(tmax), "t_min": np.nanmean(tmin),
        "t_delta": np.nanmean(tmax - tmin), "t_start": tmean[0], "t_rise": tmean[-1] - tmean[0],
        "i_mean": np.nanmean(i), "i_max": np.nanmax(i),
        "spread_mean": np.nanmean(spread), "spread_max": np.nanmax(spread),
        "spread_end": np.nanmedian(spread[-10:]),
        "cellv_max_end": np.nanmedian(seg["max_cell_voltage"].to_numpy(float)[-10:]),
        "v_start": np.nanmedian(v[:5]), "v_end": np.nanmedian(v[-10:]),
    }
    for p in SOC_POINTS:
        m = np.abs(soc - p) <= HALF_BIN
        if m.sum() >= MIN_SAMPLES_BIN:
            f[f"v@{p}"] = np.nanmedian(v[m]); f[f"i@{p}"] = np.nanmedian(i[m])
            f[f"spread@{p}"] = np.nanmedian(spread[m]); f[f"t@{p}"] = np.nanmedian(tmean[m])
        else:
            f[f"v@{p}"] = f[f"i@{p}"] = f[f"spread@{p}"] = f[f"t@{p}"] = np.nan
    for lo, hi in SOC_WINDOWS:
        m = (soc >= lo) & (soc <= hi)
        if m.sum() >= 20 and np.ptp(soc[m]) >= 0.5 * (hi - lo):
            f[f"dVdSOC@{lo}-{hi}"] = np.polyfit(soc[m], v[m], 1)[0]
        else:
            f[f"dVdSOC@{lo}-{hi}"] = np.nan
    return f


def process_vehicle(vid):
    df = pd.read_csv(RAW_DIR / f"#{vid}.csv")
    df.columns = COLS
    df = df.sort_values("record_time").reset_index(drop=True)
    rows = []
    for seg in find_segments(df):
        seg = seg.reset_index(drop=True)
        if len(seg) < 100:
            continue
        d = seg["soc"].diff().iloc[1:]
        if (d > 2).any() or (d < -0.1).any():
            continue
        cap = capacity_ah(seg)
        if cap == 0:
            continue
        r = {"vehicle": vid, "time_start": pd.to_datetime(str(seg["record_time"].iloc[0])), "capacity_recalc": cap}
        r.update(session_features(seg))
        rows.append(r)
    return pd.DataFrame(rows)


def add_rolling(d):
    """차량별 인과적(trailing) 롤링 중앙값: 현재 세션 + 과거 세션만 사용"""
    key = ["v@50", "v@70", "dVdSOC@30-60", "dVdSOC@60-90", "spread_mean", "spread_end", "t_mean", "i_mean"]
    d = d.sort_values(["vehicle", "time_start"]).copy()
    for k in key:
        d[f"{k}_r{ROLL}"] = d.groupby("vehicle")[k].transform(
            lambda s: s.rolling(ROLL, min_periods=3).median())
    return d


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    vids = list(range(1, 21))
    parts = Parallel(n_jobs=6)(delayed(process_vehicle)(v) for v in vids)
    feats = pd.concat(parts, ignore_index=True)
    print(f"recomputed sessions: {len(feats)}")

    # 1) 세션 재현 검증: 용량이 processed 표와 일치하는가
    ref = pd.read_csv(PROCESSED)
    ref["vehicle"] = ref["vehicle_id"].astype(int)
    assert len(ref) == len(feats), f"session count mismatch {len(ref)} vs {len(feats)}"
    ref = ref.sort_values(["vehicle", "time_start"]).reset_index(drop=True)
    feats = feats.sort_values(["vehicle", "time_start"]).reset_index(drop=True)
    diff = np.abs(ref["capacity_ah"].to_numpy() - feats["capacity_recalc"].to_numpy()).max()
    assert diff < 1e-6, f"capacity mismatch {diff}"
    print(f"session reproduction OK (max capacity diff {diff:.2e})")

    feats = add_rolling(feats)

    # 2) 통합본 BatICM 행과 순서 정렬로 대응 (팀이 일부 세션을 제외했으므로 부분수열 매칭)
    m = pd.read_pickle(MERGED_CACHE)
    b = m[m["source_dataset"] == "BatICM"].copy()
    b["vehicle"] = b["entity_id"].str.replace("BatICM_", "").astype(int)
    out = []
    for v in vids:
        fv = feats[feats.vehicle == v].reset_index(drop=True)
        bv = b[b.vehicle == v].sort_values("cycle_index")
        fk = fv["capacity_recalc"].to_numpy()
        j, idx = 0, []
        for cap in bv["capacity_ah_unified"].to_numpy():
            while j < len(fk) and abs(fk[j] - cap) > 1e-5:
                j += 1
            assert j < len(fk), f"vehicle {v}: no alignment"
            idx.append(j); j += 1
        sel = fv.iloc[idx].reset_index(drop=True)
        sel.insert(0, "row_id", bv["row_id"].to_numpy())
        # 대응 검증: 경과일이 통합본과 일치하는가 (첫 행 기준 상대값)
        d_raw = (sel["time_start"] - sel["time_start"].iloc[0]).dt.total_seconds() / 86400
        d_mrg = bv["time_elapsed_days"].to_numpy() - bv["time_elapsed_days"].iloc[0]
        assert np.abs(d_raw.to_numpy() - d_mrg).max() < 0.01, f"vehicle {v}: time misaligned"
        out.append(sel)
    out = pd.concat(out, ignore_index=True)
    assert out["row_id"].is_unique and len(out) == len(b)
    drop = ["capacity_recalc", "time_start", "vehicle"]
    out.drop(columns=drop).to_pickle(OUT)
    fcols = [c for c in out.columns if c not in ["row_id"] + drop]
    print(f"aligned {len(out)} rows -> {OUT.name} | {len(fcols)} feature columns")
    print("NaN rate:\n" + out[fcols].isna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
