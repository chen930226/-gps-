# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
from datetime import datetime
import math

# -------------------------------
# 固定檔名設定（小寫）
moving_csv = "gps_imu_data2.csv"
calib_json = "calibration.json"
out_csv = "filtered.csv"
# -------------------------------

# 讀取 CSV
df = pd.read_csv(moving_csv)

# 讀取校正 JSON
with open(calib_json, "r") as f:
    calib = json.load(f)

lat0 = calib.get("lat0", 0.0)
lon0 = calib.get("lon0", 0.0)
R_matrix = np.array(calib.get("R_2x2_m2", [[25,0],[0,25]]))

# 經緯度 ↔ EN
def meters_per_deg(lat_deg):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon

def ll_to_en(lat, lon, lat0, lon0):
    mlat, mlon = meters_per_deg(lat0)
    e = (lon - lon0) * mlon
    n = (lat - lat0) * mlat
    return e, n

def en_to_ll(e, n, lat0, lon0):
    mlat, mlon = meters_per_deg(lat0)
    lat = n / mlat + lat0
    lon = e / mlon + lon0
    return lat, lon

# -------------------------------
# 簡單 GPSCorrector 類別，加入衛星數 sat 權重
class GPSCorrector:
    def __init__(self, lat0, lon0, R, q_acc=0.8, speed_limit_mps=70.0, r_gate=50.0):
        self.lat0 = lat0
        self.lon0 = lon0
        self.R = np.array(R, dtype=float)
        self.q_acc = q_acc
        self.speed_limit = speed_limit_mps
        self.r_gate = r_gate
        self.x = None
        self.P = None
        self.last_ts = None
        self.H = np.array([[1,0,0,0],[0,1,0,0]])

    def _F_Q(self, dt):
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        G = np.array([[0.5*dt*dt,0],[0,0.5*dt*dt],[dt,0],[0,dt]])
        Q = (self.q_acc**2) * (G @ G.T)
        return F, Q

    def _fuse_sources(self, A=None, B=None, C=None):
        cand = []
        for src in (A,B,C):
            if not src: continue
            la, lo = src.get("lat"), src.get("lon")
            Rm = src.get("R", None)
            sat = src.get("Sat", 1)
            if la is None or lo is None: continue
            if (Rm is not None) and (Rm > self.r_gate): continue
            cand.append((la, lo, Rm, sat))
        if not cand: return None, None
        wsum=0.0; lat_w=0.0; lon_w=0.0
        for (la,lo,Rm,sat) in cand:
            if Rm is not None and Rm > 0:
                w = sat / (Rm*Rm)
            else:
                w = sat
            wsum += w; lat_w += w*la; lon_w += w*lo
        return lat_w/wsum, lon_w/wsum

    def update(self, ts=None, A=None, B=None, C=None):
        fused_lat, fused_lon = self._fuse_sources(A,B,C)
        if fused_lat is None:
            return None
        e, n = ll_to_en(fused_lat, fused_lon, self.lat0, self.lon0)
        if ts is None:
            ts = datetime.now().timestamp()
        dt = 0.2 if self.last_ts is None else max(0.02, float(ts - self.last_ts))
        self.last_ts = ts
        F, Q = self._F_Q(dt)
        do_update = True
        if self.x is not None:
            de = e - float(self.x[0,0])
            dn = n - float(self.x[1,0])
            inst_v = math.hypot(de,dn)/max(dt,1e-3)
            if inst_v > self.speed_limit:
                do_update = False
        if self.x is None:
            self.x = np.array([[e],[n],[0.0],[0.0]])
            self.P = np.eye(4)*10
        else:
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q
        if do_update:
            z = np.array([[e],[n]])
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            y = z - self.H @ self.x
            self.x = self.x + K @ y
            I = np.eye(4)
            self.P = (I - K @ self.H) @ self.P
        est_e, est_n = float(self.x[0,0]), float(self.x[1,0])
        corr_lat, corr_lon = en_to_ll(est_e, est_n, self.lat0, self.lon0)
        return corr_lat, corr_lon, fused_lat, fused_lon

# -------------------------------
# 建立 Corrector
corrector = GPSCorrector(lat0, lon0, R_matrix)

# 假設 CSV 有 A/B/C GPS 欄位 + Sat 欄位
corrected_list = []
for i,row in df.iterrows():
    A = {'lat':row["A_Lat"],'lon':row["A_Lon"],'R':row["A_R"],'Sat':row["A_Sat"]} if not pd.isna(row["A_Lat"]) else None
    B = {'lat':row["B_Lat"],'lon':row["B_Lon"],'R':row["B_R"],'Sat':row["B_Sat"]} if not pd.isna(row["B_Lat"]) else None
    C = {'lat':row["C_Lat"],'lon':row["C_Lon"],'R':row["C_R"],'Sat':row["C_Sat"]} if not pd.isna(row["C_Lat"]) else None
    out = corrector.update(ts=None, A=A, B=B, C=C)
    if out is None:
        corrected_list.append((np.nan,np.nan,np.nan,np.nan))
    else:
        corrected_list.append(out)

df[["Corrected_Lat","Corrected_Lon","Fused_Lat_Calc","Fused_Lon_Calc"]] = pd.DataFrame(corrected_list)

# 儲存結果
df.to_csv(out_csv,index=False)
print(f"✅ 已完成濾波與校正，輸出檔案：{out_csv}")
