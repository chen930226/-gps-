# -*- coding: utf-8 -*-
import json, math
import pandas as pd
import numpy as np

# ===== 檔名設定 =====
INPUT_CSV = "gps_imu_data.csv"
OUT_JSON  = "calibration.json"

# ===== 參數（可依模組微調）=====
UERE_M_DEFAULT = 5.0     # 由 HDOP 估半徑的 UERE（一般 NEO-6M/8N 開闊地 ~4~6m）
MIN_SAT = 4              # 少於 4 顆衛星的解通常不穩，直接跳過
MAX_DEG_ABS = 90.0       # 緯度絕對值上限（基本 sanity check）
# 若你的 CSV 有 A_R/B_R/C_R 則優先用；沒有的話會用 HDOP*UERE 估 r
# 欄位假定（可依你的實際欄位名調整）：
COLS = {
    "A": {"lat":"A_Lat","lon":"A_Lon","r":"A_R","hdop":"A_HDOP","sat":"A_Sat"},
    "B": {"lat":"B_Lat","lon":"B_Lon","r":"B_R","hdop":"B_HDOP","sat":"B_Sat"},
    "C": {"lat":"C_Lat","lon":"C_Lon","r":"C_R","hdop":"C_HDOP","sat":"C_Sat"},
}

def meters_per_deg(lat_deg: float):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon

def ll_to_en_local(lat, lon, lat_ref, lon_ref):
    """以該筆 fused 緯度作比例，將(經緯)相對於(ref_lat, ref_lon)轉公尺"""
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
    """w = Sat / r^2（無 r 則 w = Sat；無 Sat 則 w = 1），回傳 (fused_lat, fused_lon, used_count)"""
    lats, lons, ws = [], [], []
    used = 0
    for k in keys:
        lat = robust_float(row.get(COLS[k]["lat"], None))
        lon = robust_float(row.get(COLS[k]["lon"], None))
        if lat is None or lon is None:
            continue
        if abs(lat) > MAX_DEG_ABS or abs(lon) > 180:
            continue
        sat = get_sat(row, k)
        r = derive_r(row, k)
        if sat is None and r is None:
            w = 1.0
        elif r is None:
            w = max(sat, 1.0)
        elif sat is None:
            w = 1.0 / max(r*r, 1e-6)
        else:
            w = max(sat, 1.0) / max(r*r, 1e-6)
        # 可選：過濾低衛星數
        if sat is not None and sat < MIN_SAT:
            continue
        lats.append(lat); lons.append(lon); ws.append(w); used += 1

    if used == 0:
        return None, None, 0
    wsum = sum(ws)
    fused_lat = sum(w*lat for w,lat in zip(ws, lats)) / wsum
    fused_lon = sum(w*lon for w,lon in zip(ws, lons)) / wsum
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
            e, n = ll_to_en_local(lat, lon, fused_lat, fused_lon)  # 感測器位置 - fused 位置
            # 收進總殘差池
            res_all_e.append(e); res_all_n.append(n)
            # 分別紀錄到每顆的池
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
            "weight_rule": "w = Sat / r^2; r from R or (HDOP*UERE)"
        }
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("✅ 已輸出", OUT_JSON)
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
