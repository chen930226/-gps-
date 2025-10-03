# -*- coding: utf-8 -*-
import serial
import threading
import time
import csv
from datetime import datetime
import os
import struct
import json
import math
from typing import Optional, Dict, Tuple
import numpy as np

baud_rate = 115200
output_csv = "gps_imu_data2.csv"
CSV_FILE = output_csv

CALIB_JSON = "calibration.json"
Q_ACC = 0.8
SPEED_LIMIT_MPS = 70.0
R_GATING_M = 50.0      

def load_com_map(config_file="com_mapping_config.txt"):
    com_map = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                if ':' in line:
                    com, role = line.strip().split(':')
                    com_map[com] = role
        roles = set(com_map.values())
        if roles != {"A","B","C"}:
            raise ValueError(f"COM ERROR {roles}")
    except Exception as e:
        exit(1)
    return com_map

COM_MAP = load_com_map()
arduino_COM = list(COM_MAP.keys())

def meters_per_deg(lat_deg: float) -> Tuple[float,float]:
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82*math.cos(2*lat) + 1.175*math.cos(4*lat) - 0.0023*math.cos(6*lat)
    m_per_deg_lon = 111412.84*math.cos(lat) - 93.5*math.cos(3*lat) + 0.118*math.cos(5*lat)
    return m_per_deg_lat, m_per_deg_lon

def ll_to_en(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float,float]:
    mlat, mlon = meters_per_deg(lat0)
    e = (lon - lon0) * mlon
    n = (lat - lat0) * mlat
    return e, n

def en_to_ll(e: float, n: float, lat0: float, lon0: float) -> Tuple[float,float]:
    mlat, mlon = meters_per_deg(lat0)
    lat = n / mlat + lat0
    lon = e / mlon + lon0
    return lat, lon

class GPSCorrector:
    def __init__(self, lat0, lon0, R_2x2_m2,
                 q_acc=0.8, speed_limit_mps=70.0, r_gating_m=50.0):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        self.R = np.array(R_2x2_m2, dtype=float)
        self.q_acc = float(q_acc)
        self.speed_limit = float(speed_limit_mps)
        self.r_gate = float(r_gating_m)

        self.x = None
        self.P = None 
        self.last_ts = None
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)

    @classmethod
    def from_json(cls, path: str, q_acc=0.8, speed_limit_mps=70.0, r_gating_m=50.0):
        if os.path.exists(path):
            with open(path, "r") as f:
                d = json.load(f)
            print("✅ calibration.json 已載入")

            if "lat0" in d and "lon0" in d and "R_2x2_m2" in d:
                return cls(d["lat0"], d["lon0"], d["R_2x2_m2"],
                        q_acc=q_acc, speed_limit_mps=speed_limit_mps, r_gating_m=r_gating_m)
            elif "R_sensor_m2" in d:
                return cls(24.15, 120.70, d["R_sensor_m2"],
                        q_acc=q_acc, speed_limit_mps=speed_limit_mps, r_gating_m=r_gating_m)
            else:
                raise KeyError("calibration.json 缺少必要欄位 (lat0/lon0/R_2x2_m2 或 R_sensor_m2)")
        else:
            print("⚠️ 找不到 calibration.json，使用預設值")
            return cls(24.0, 120.0, [[25.0, 0.0], [0.0, 25.0]],
                    q_acc=q_acc, speed_limit_mps=speed_limit_mps, r_gating_m=r_gating_m)

    def _F_Q(self, dt: float):
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        G = np.array([[0.5*dt*dt,0],[0,0.5*dt*dt],[dt,0],[0,dt]], dtype=float)
        Q = (self.q_acc**2) * (G @ G.T)
        return F, Q

    def _fuse_sources(self, A: Optional[Dict], B: Optional[Dict], C: Optional[Dict]):
        cand = []
        for src in (A,B,C):
            if not src: continue
            la, lo = src.get("lat"), src.get("lon")
            Rm = src.get("R", None)
            sat = src.get("Sat", 1)
            if la is None or lo is None: continue
            if (Rm is not None) and (Rm > self.r_gate): continue
            cand.append((la,lo,Rm,sat))
        if not cand: return None, None
        wsum=0.0; lat_w=0.0; lon_w=0.0
        for (la,lo,Rm,sat) in cand:
            if Rm is not None and Rm > 0:
                w = sat / (Rm*Rm)
            else:
                w = sat
            wsum += w; lat_w += w*la; lon_w += w*lo
        return lat_w/wsum, lon_w/wsum

    def update(self, ts: Optional[float], A: Optional[Dict]=None, B: Optional[Dict]=None, C: Optional[Dict]=None):
        fused_lat, fused_lon = self._fuse_sources(A,B,C)
        if fused_lat is None:
            return None

        e, n = ll_to_en(fused_lat, fused_lon, self.lat0, self.lon0)

        if ts is None:
            ts = time.time()
        dt = 0.2 if self.last_ts is None else max(0.02, float(ts - self.last_ts))
        self.last_ts = ts

        F, Q = self._F_Q(dt)

        do_update = True
        if self.x is not None:
            de = e - float(self.x[0,0])
            dn = n - float(self.x[1,0])
            inst_v = math.hypot(de, dn) / max(dt, 1e-3)
            if inst_v > self.speed_limit:
                do_update = False

        if self.x is None:
            self.x = np.array([[e],[n],[0.0],[0.0]], dtype=float)
            self.P = np.eye(4)*10.0
        else:
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        if do_update:
            z = np.array([[e],[n]], dtype=float)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            y = z - (self.H @ self.x)
            self.x = self.x + K @ y
            I = np.eye(4)
            self.P = (I - K @ self.H) @ self.P

        est_e = float(self.x[0,0]); est_n = float(self.x[1,0])
        corr_lat, corr_lon = en_to_ll(est_e, est_n, self.lat0, self.lon0)
        return (corr_lat, corr_lon, fused_lat, fused_lon)

COM_GPS_data = {port: None for port in arduino_COM}
stop_flag = False
data_lock = threading.Lock()
count = 0

def read_gps(port):
    global COM_GPS_data
    try:
        ser = serial.Serial(port, baud_rate, timeout=0.5)
        while not stop_flag:
            if ser.in_waiting:
                data = ser.readline().decode(errors='ignore').strip()
                if 'ID' not in data:
                    try:
                        parts = data.split(',')
                        if len(parts) >= 4:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            r = float(parts[2])
                            sat = int(parts[3])
                            with data_lock:
                                COM_GPS_data[port] = [lat, lon, r, sat]
                    except ValueError:
                        print(f"[{port}] 無法解析資料：{data}")
    except Exception as e:
        print(f"[{port}] 錯誤: {e}")

def read_imu():
    imu_ser = serial.Serial('COM11', 115200, timeout=0.5)
    accum_acc = [0.0, 0.0, 0.0]
    accum_gyro = [0.0, 0.0, 0.0]
    accum_count = 0
    roll = pitch = yaw = None
    last_time = time.time()

    while not stop_flag:
        if imu_ser.in_waiting >= 1:
            if imu_ser.read() == b'\x55':
                head = imu_ser.read(1)
                raw = imu_ser.read(8)
                if len(raw) != 8:
                    continue
                buf = struct.unpack('<8B', raw)
                dtype = buf[0]

                if dtype == 0x01:
                    roll = (buf[2] << 8 | buf[1]) / 32768.0 * 180
                    pitch = (buf[4] << 8 | buf[3]) / 32768.0 * 180
                    yaw = (buf[6] << 8 | buf[5]) / 32768.0 * 180
                elif dtype == 0x03:
                    ax = (buf[2] << 8 | buf[1]) / 32768.0 * 2 * 9.8
                    ay = (buf[4] << 8 | buf[3]) / 32768.0 * 2 * 9.8
                    az = (buf[6] << 8 | buf[5]) / 32768.0 * 2 * 9.8
                    gx = (buf[2] << 8 | buf[1]) / 32768.0 * 90
                    gy = (buf[4] << 8 | buf[3]) / 32768.0 * 90
                    gz = (buf[6] << 8 | buf[5]) / 32768.0 * 90
                    accum_acc[0] += ax
                    accum_acc[1] += ay
                    accum_acc[2] += az
                    accum_gyro[0] += gx
                    accum_gyro[1] += gy
                    accum_gyro[2] += gz
                    accum_count += 1

        # ← 改成每 0.2 秒存一次平均值
        if time.time() - last_time >= 0.2 and accum_count > 0:
            avg_acc = [v / accum_count for v in accum_acc]
            avg_gyro = [v / accum_count for v in accum_gyro]

            with data_lock:
                imu_data_queue.append((
                    f"{roll:.2f}", f"{pitch:.2f}", f"{yaw:.2f}",
                    *["{:.2f}".format(x) for x in avg_acc],
                    *["{:.2f}".format(x) for x in avg_gyro]
                ))

            accum_acc = [0.0, 0.0, 0.0]
            accum_gyro = [0.0, 0.0, 0.0]
            accum_count = 0
            last_time = time.time()

EXPECTED_HEADER = [
    'A_Lat', 'A_Lon', 'A_R', 'A_Sat',
    'B_Lat', 'B_Lon', 'B_R', 'B_Sat',
    'C_Lat', 'C_Lon', 'C_R', 'C_Sat',
    'Fused_Lat', 'Fused_Lon',
    'Corrected_Lat', 'Corrected_Lon',
    'Roll', 'Pitch', 'Yaw',
    'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz',
    'Timestamp'
]

def ensure_csv_header(path: str, header: list):
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as f:
            csv.writer(f).writerow(header)
        print(f"🆕 已建立 CSV 並寫入新表頭：{path}")
        return

    try:
        with open(path, 'r', newline='') as f:
            first = f.readline()
        if first:
            old_header = [c.strip() for c in first.strip().split(',')]
            if old_header != header:
                backup = path + ".old"
                os.rename(path, backup)
                with open(path, 'w', newline='') as f:
                    csv.writer(f).writerow(header)
                print(f"CSV 欄位不同，已備份為 {backup} 並建立新檔 {path}")
    except Exception as e:
        print(f" CSV 頭發生例外：{e}")

ensure_csv_header(CSV_FILE, EXPECTED_HEADER)

imu_data_queue = []
threading.Thread(target=read_imu, daemon=True).start()
for port in arduino_COM:
    threading.Thread(target=read_gps, args=(port,), daemon=True).start()

corrector = GPSCorrector.from_json(CALIB_JSON, q_acc=Q_ACC, speed_limit_mps=SPEED_LIMIT_MPS, r_gating_m=R_GATING_M)

try:
    while count < 18000:
        time.sleep(0.2)   # ← 改成每 0.2 秒收集一次
        with data_lock:
            ready_gps = all(COM_GPS_data[port] is not None for port in arduino_COM)
            if ready_gps and imu_data_queue:
                gps_order = ['A', 'B', 'C']
                ordered = [COM_GPS_data[port] for port, role in sorted(COM_MAP.items(), key=lambda x: gps_order.index(x[1]))]
                imu_values = imu_data_queue.pop(0)

                A = {'lat': ordered[0][0], 'lon': ordered[0][1], 'R': ordered[0][2], 'Sat': ordered[0][3]} if len(ordered) > 0 else None
                B = {'lat': ordered[1][0], 'lon': ordered[1][1], 'R': ordered[1][2], 'Sat': ordered[1][3]} if len(ordered) > 1 else None
                C = {'lat': ordered[2][0], 'lon': ordered[2][1], 'R': ordered[2][2], 'Sat': ordered[2][3]} if len(ordered) > 2 else None

                ts = datetime.now().timestamp()
                out = corrector.update(ts, A=A, B=B, C=C)

                if out is None:
                    for port in arduino_COM:
                        COM_GPS_data[port] = None
                    continue

                corrected_lat, corrected_lon, fused_lat, fused_lon = out

                print(f"收集到 GPS 資料（A/B/C）：{ordered}")
                print(f"Fused = ({fused_lat:.7f},{fused_lon:.7f}) / Corrected = ({corrected_lat:.7f},{corrected_lon:.7f})")
                print(f"IMU = {imu_values}")

                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = []
                    for data in ordered:
                        row.extend(data)
                    row.extend([fused_lat, fused_lon, corrected_lat, corrected_lon])
                    row.extend(imu_values)
                    row.append(datetime.now().isoformat())
                    writer.writerow(row)

                count += 1
                print(f"✅ 已寫入第 {count} 筆資料")

                for port in arduino_COM:
                    COM_GPS_data[port] = None

    stop_flag = True
    print("已收集滿資料，自動結束")

except KeyboardInterrupt:
    stop_flag = True
    print("結束收集")
