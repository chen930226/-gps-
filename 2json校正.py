# -*- coding: utf-8 -*-
import json, math
import pandas as pd
import numpy as np

# ===== 檔名設定 =====
INPUT_CSV = "gps_imu_data.csv"   # 確認與你的寫檔檔名相同
OUT_JSON  = "calibration.json"

# ===== 參數（可依模組微調）=====
UERE_M_DEFAULT = 5.0     # 由 HDOP 估半徑的 UERE（NEO-6/8 開闊地 ~4~6m）
MIN_SAT = 4              # 少於 4 顆衛星的解通常不穩，直接跳過
MAX_DEG_ABS = 90.0       # 緯度絕對值上限（基本 sanity check）

# === 依照你目前 CSV 欄位名稱更新對應 ===
# 若沒有 A_R/B_R/C_R 會自動 fallback 用 HDOP*UERE 估 r
COLS = {
    "A": {"lat":"A_Lat","lon":"A_Lon","r":"A_R","hdop":"A_HDOP","sat":"A_SatUsed","nsat":"A_NSatView"},
    "B": {"lat":"B_Lat","lon":"B_Lon","r":"B_R","hdop":"B_HDOP","sat":"B_SatUsed","nsat":"B_NSatView"},
    "C": {"lat":"C_Lat","lon":"C_Lon","r":"C_R","hdop":"C_HDOP","sat":"C_SatUsed","nsat":"C_NSatView"},

}

def meters_per_deg(lat_deg: float):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon

def ll_to_en_local(lat, lon, lat_ref, lon_ref):
    """以該筆 fused 緯度做比例，將(經緯)相對於(ref_lat, ref_lon)轉公尺"""
    mlat, mlon = meters_per_deg(lat_ref)
    e = (lon - lon_ref) * mlon
    n = (lat - lat_ref) * mlat
    return e, n

def robust_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except:
        return None

def derive_r(row, key):
    """優先用 R 欄位；沒有就用 HDOP*UERE 推估 r。若取不到回 None。"""
    conf = COLS[key]
    r = None
    if conf["r"] in row and pd.notna(row[conf["r"]]):
        r = robust_float(row[conf["r"]])
        if r is not None and r > 0:
            return r
    # fallback: HDOP * UERE
    hdop = None
    if conf["hdop"] in row and pd.notna(row[conf["hdop"]]):
        hdop = robust_float(row[conf["hdop"]])
    if hdop is not None and hdop > 0:
        return hdop * UERE_M_DEFAULT
    return None

def get_sat(row, key):
    conf = COLS[key]
    if conf["sat"] in row and pd.notna(row[conf["sat"]]):
        s = robust_float(row[conf["sat"]])
        if s is not None and s > 0:
            return s
    return None

def weighted_fuse_row(row, keys=("A","B","C")):

    lats, lons, ws = [], [], []
    used = 0
    for k in keys:
        lat = robust_float(row.get(COLS[k]["lat"], None))
        lon = robust_float(row.get(COLS[k]["lon"], None))
        if lat is None or lon is None:
            continue
        if abs(lat) > MAX_DEG_ABS or abs(lon) > 180:
            continue

        # 既有取得：sat、r（可能為 None）
        sat = get_sat(row, k)
        r = derive_r(row, k)

        # 新增：取 hdop/nsat/snr（容忍缺值）
        hdop = robust_float(row.get(COLS[k]["hdop"], None))
        nsat = robust_float(row.get(COLS[k].get("nsat", ""), None))

        # 有效值處理（缺值時採保守預設）
        sat_eff  = max(sat, 1.0) if sat is not None else 1.0
        nsat_eff = max(nsat, 0.0) if nsat is not None else 0.0

        if r is None:
            if hdop is not None and hdop > 0:
                r_eff = hdop * UERE_M_DEFAULT
            else:
                r_eff = 10.0  # 無資訊時的保守半徑（與原版語意一致）
        else:
            r_eff = max(r, 1e-3)


        # 綜合權重（其餘流程與篩選規則保持不變）
        if sat is not None and sat < MIN_SAT:
            continue
        denom = r_eff 
        w = (sat_eff + 0.5 * nsat_eff) / max(denom * denom, 1e-6)

        lats.append(lat); lons.append(lon); ws.append(w); used += 1

    if used == 0:
        return None, None, 0
    wsum = sum(ws)
    fused_lat = sum(w*lat for w, lat in zip(ws, lats)) / wsum
    fused_lon = sum(w*lon for w, lon in zip(ws, lons)) / wsum
    return fused_lat, fused_lon, used

def main():
    df = pd.read_csv(INPUT_CSV)

    # 計殘差（A/B/C 相對 fused）
    res_all_e, res_all_n = [], []
    res_A_e, res_A_n = [], []
    res_B_e, res_B_n = [], []
    res_C_e, res_C_n = [], []

    total_rows = 0
    used_rows  = 0

    for _, row in df.iterrows():
        total_rows += 1
        fused_lat, fused_lon, used_cnt = weighted_fuse_row(row)
        if fused_lat is None or fused_lon is None:
            continue
        used_rows += 1

        # 以該筆 fused 當「臨時參考」
        for k in ("A","B","C"):
            lat = robust_float(row.get(COLS[k]["lat"], None))
            lon = robust_float(row.get(COLS[k]["lon"], None))
            if lat is None or lon is None:
                continue
            if abs(lat) > MAX_DEG_ABS or abs(lon) > 180:
                continue
            e, n = ll_to_en_local(lat, lon, fused_lat, fused_lon)
            res_all_e.append(e); res_all_n.append(n)
            if k == "A":
                res_A_e.append(e); res_A_n.append(n)
            elif k == "B":
                res_B_e.append(e); res_B_n.append(n)
            else:
                res_C_e.append(e); res_C_n.append(n)

    def cov2x2(es, ns):
        if len(es) >= 2 and len(ns) >= 2:
            return np.cov(np.vstack([np.array(es, dtype=float), np.array(ns, dtype=float)])).tolist()
        return None

    R_sensor_all = cov2x2(res_all_e, res_all_n)
    R_A = cov2x2(res_A_e, res_A_n)
    R_B = cov2x2(res_B_e, res_B_n)
    R_C = cov2x2(res_C_e, res_C_n)

    out = {
        "method": "per-sample_residuals_of_A/B/C_to_fused_without_truth",
        "samples_total_rows": total_rows,
        "samples_used_rows": used_rows,
        "samples_residuals_all": len(res_all_e),
        "R_sensor_m2": R_sensor_all,  # 建議當作卡曼濾波量測噪聲 R
        "R_A_m2": R_A,                # 單顆（可選）
        "R_B_m2": R_B,
        "R_C_m2": R_C,
        "params": {
            "UERE_m_default": UERE_M_DEFAULT,
            "min_sat": MIN_SAT,
           "weight_rule": "w = (SatUsed + 0.5*NSatView) / (r_eff^2); r_eff from R or (HDOP*UERE)"

        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("✅ 已輸出", OUT_JSON)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
