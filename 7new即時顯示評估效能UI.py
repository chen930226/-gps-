# -*- coding: utf-8 -*-
# ✅ GPS 即時定位 GUI（Raw vs Processed 比較 + CSV紀錄 + 結束後折線圖）
# - 參考狀態（手動/自動）
# - Raw vs Processed 漂移、CEP50/95、改善%
# - 效能圖（GPS/IMU/System/Inference）
# - 即時寫 ab_eval.csv
# - GUI 關閉後自動畫折線圖（Raw/Processed 漂移、CEP95、改善%）
#
# 注意：本檔保持你原本流程（map.html 不變），在右側新增比較 + 效能圖 + 紀錄功能。

import sys, os, threading, struct, time, csv
import numpy as np
import torch
import serial
from collections import deque
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
                             QGroupBox, QGridLayout, QPushButton)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import Qt, QUrl, QTimer
from math import radians, cos, degrees
from datetime import datetime
from scipy.optimize import least_squares
import joblib
import logging  
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ------------------------- 參數 -------------------------
EARTH_RADIUS = 6371000
COM_PORTS = ['COM7', 'COM8', 'COM9']
IMU_PORT = 'COM11'
BAUD_RATE = 115200
MODEL_PATH = "best_gps_model.pth"
SCALER_PATH = "scaler.pkl"
UPDATE_INTERVAL_MS = 200          # 系統更新率 ≈ 5Hz
WINDOW = 5                        # GeoMLP 時間窗
IMU_WINDOW = 5                    # IMU 統計窗

AUTO_REF_WINDOW_SEC = 10          # 自動參考（最近 N 秒 KF 中位數）
STATS_WINDOW_SEC = 60             # CEP 統計窗口（最近 N 秒）
PROCESSED_KIND = "KF"             # "KF" or "FUSED" 作為處理後輸出

CSV_FILE = "ab_eval.csv"          # 紀錄檔名

# ------------------------- 座標轉換 -------------------------
def latlon_to_xy(lat, lon, origin_lat):
    x = radians(lon) * EARTH_RADIUS * cos(radians(origin_lat))
    y = radians(lat) * EARTH_RADIUS
    return x, y

def xy_to_latlon(x, y, origin_lat):
    lat = degrees(y / EARTH_RADIUS)
    lon = degrees(x / (EARTH_RADIUS * cos(radians(origin_lat))))
    return lat, lon

# ------------------------- GeoMLP -------------------------
class GeoMLP(torch.nn.Module):
    def __init__(self, input_dim=51*WINDOW):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2)  # Δx, Δy
        )
    def forward(self, x):
        return self.net(x)

# ------------------------- Kalman Filter -------------------------
class KalmanFilter2D:
    def __init__(self):
        self.x = np.zeros(4)           # [x, y, vx, vy]
        self.P = np.eye(4) * 1.0
        self.F = np.eye(4)
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.R = np.eye(2) * 5.0
        self.Q = np.eye(4) * 0.01
    def predict(self, dt=0.2):
        self.F[0,2] = dt
        self.F[1,3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.x[:2]

# ------------------------- 串口讀取 -------------------------
class SerialReader(threading.Thread):
    def __init__(self, port, shared_dict, timestamp_dict):
        super().__init__(daemon=True)
        self.port = port
        self.ser = None
        self.running = False
        self.shared_dict = shared_dict
        self.timestamp_dict = timestamp_dict
    def run(self):
        try:
            self.ser = serial.Serial(self.port, BAUD_RATE, timeout=0.2)
            self.running = True
            while self.running:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if 'ID' in line:
                        continue
                    parts = line.split(',')
                    if len(parts) < 3:
                        continue
                    try:
                        if len(parts) >= 4:
                            lat, lon, r, sat = float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
                        else:
                            lat, lon, r, sat = float(parts[0]), float(parts[1]), float(parts[2]), 5
                        self.shared_dict[self.port] = [lat, lon, r, sat]
                        self.timestamp_dict[self.port] = datetime.now()
                        # 效能：GPS 計數
                        perf = self.shared_dict.setdefault('_perf', {})
                        perf['gps'] = perf.get('gps', 0) + 1
                    except:
                        continue
        except Exception as e:
            print(f"[ERROR] {self.port}: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

class IMUSerialReader(threading.Thread):
    def __init__(self, port, shared_dict):
        super().__init__(daemon=True)
        self.port = port
        self.ser = None
        self.running = False
        self.shared_dict = shared_dict
        self.history = {i: [] for i in range(9)}  # 9 維 IMU
    def run(self):
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=0.2)
            self.running = True
            while self.running:
                if self.ser.in_waiting >= 33:
                    data = self.ser.read(33)
                    if len(data) >= 33 and data[0] == 0x55 and data[1] == 0x53:
                        try:
                            imu_data = list(struct.unpack("<9f", data[2:2+4*9]))
                            for i, v in enumerate(imu_data):
                                self.history[i].append(float(v))
                                if len(self.history[i]) > IMU_WINDOW:
                                    self.history[i].pop(0)
                            self.shared_dict['IMU'] = self.history
                            # 效能：IMU 計數
                            perf = self.shared_dict.setdefault('_perf', {})
                            perf['imu'] = perf.get('imu', 0) + 1
                        except:
                            continue
        except Exception as e:
            print(f"[IMU ERROR] {self.port}: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

# ------------------------- GUI -------------------------
class GPSViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPS 即時定位（Raw vs Processed + 紀錄）")
        self.resize(1300, 780)

        # 共享與狀態
        self.shared_data = {}
        self.timestamps = {}
        self.last_processed_time = None
        self.r_history = {port: [] for port in COM_PORTS}

        # 參考點與統計
        self.ref_xy = None
        self.auto_ref_buffer = deque()      # KF 座標的時間序列，供自動參考
        self.stats_raw = deque()            # (t, dist) 最近窗口的 Raw 漂移
        self.stats_proc = deque()           # (t, dist) 最近窗口的 Processed 漂移

        # 模型與 scaler
        self.model = GeoMLP()
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            print(f"[INFO] loaded model: {MODEL_PATH}")
        except Exception as e:
            print("[WARN] model load failed:", e)
        self.model.eval()
        try:
            self.scaler = joblib.load(SCALER_PATH)
            print("[INFO] scaler.pkl loaded")
        except Exception as e:
            print("[WARN] scaler load failed:", e)
            self.scaler = None

        self.feature_buffer = deque(maxlen=WINDOW)
        self.kf = KalmanFilter2D()

        # 效能統計
        self.sys_count = 0
        self.infer_times_bucket = []
        self.perf_time0 = time.time()
        self.perf_history = {"time": deque(maxlen=300),
                             "gps": deque(maxlen=300),
                             "imu": deque(maxlen=300),
                             "sys": deque(maxlen=300),
                             "infer": deque(maxlen=300)}
        self.last_perf_snapshot = {"gps": None, "imu": None, "sys": None, "infer": None}

        # UI & 執行緒
        self.init_ui()
        self.start_serial_threads()

        # CSV 初始化
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp","Raw_Drift","Proc_Drift",
                                 "CEP50_Raw","CEP95_Raw","CEP50_Proc","CEP95_Proc",
                                 "Improve_CEP95","GPS_Hz","IMU_Hz","System_Hz","Inference_ms"])

        # 計時器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gps_data)
        self.map_view.loadFinished.connect(self.on_map_loaded)

        self.timer_perf = QTimer()
        self.timer_perf.timeout.connect(self.update_performance_plot)
        self.timer_perf.start(1000)

    # ------------------------- UI -------------------------
    def init_ui(self):
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # 影像占位
        self.image_label = QLabel("影像畫面")
        self.image_label.setFixedHeight(250)
        self.image_label.setStyleSheet("background-color: lightgray; border:1px solid black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_label)

        # 地圖
        self.map_view = QWebEngineView()
        map_path = os.path.abspath("map.html")
        self.map_view.load(QUrl.fromLocalFile(map_path))
        left_layout.addWidget(self.map_view)

        # 右側區塊
        right_layout = QVBoxLayout()

        # (1) 座標資訊
        info_box = QGroupBox("座標資訊")
        grid = QGridLayout()
        self.label_gps = QLabel("原始GPS座標(紅色): 等待中")
        self.label_geometry = QLabel("幾何計算座標(紫色): 等待中")
        self.label_mlp = QLabel("MLP預測座標(綠色): 等待中")
        self.label_fused = QLabel("融合座標(藍色): 等待中")
        self.label_kf = QLabel("Kalman Filter座標(橙色): 等待中")
        self.label_sats = QLabel("各GPS衛星數: 等待中")
        for lbl, color in [(self.label_gps,"red"),(self.label_geometry,"purple"),
                           (self.label_mlp,"green"),(self.label_fused,"blue"),
                           (self.label_kf,"orange"),(self.label_sats,"brown")]:
            lbl.setStyleSheet(f"color:{color}; font-size:16px;")
        grid.addWidget(self.label_gps, 0, 0)
        grid.addWidget(self.label_geometry, 1, 0)
        grid.addWidget(self.label_mlp, 2, 0)
        grid.addWidget(self.label_fused, 3, 0)
        grid.addWidget(self.label_kf, 4, 0)
        grid.addWidget(self.label_sats, 5, 0)
        info_box.setLayout(grid)
        right_layout.addWidget(info_box)

        # (2) Raw vs Processed 相對準確性
        ab_box = QGroupBox(f"無真值 → Raw vs 處理後（{PROCESSED_KIND}）相對準確性")
        ab_grid = QGridLayout()
        self.btn_set_ref = QPushButton("設為參考點（KF）")
        self.btn_clear_ref = QPushButton("清除參考")
        self.btn_set_ref.clicked.connect(self.on_set_reference)
        self.btn_clear_ref.clicked.connect(self.on_clear_reference)
        self.label_ref_status = QLabel(f"參考狀態：未設定（自動使用最近 {AUTO_REF_WINDOW_SEC}s KF 中位數）")
        self.label_ref_status.setStyleSheet("color:#444;")

        self.label_drift_raw  = QLabel("Raw 即時漂移: -")
        self.label_drift_proc = QLabel("Processed 即時漂移: -")
        self.label_stats_win  = QLabel(f"統計窗口：最近 {STATS_WINDOW_SEC}s")
        self.label_cep_raw    = QLabel("Raw CEP50/CEP95: - / - m")
        self.label_cep_proc   = QLabel("Processed CEP50/CEP95: - / - m")
        self.label_improve_cep95 = QLabel("改善（CEP95）：-")

        for lbl in [self.label_drift_raw, self.label_drift_proc, self.label_stats_win,
                    self.label_cep_raw, self.label_cep_proc, self.label_improve_cep95]:
            lbl.setStyleSheet("font-size:15px;")

        ab_grid.addWidget(self.btn_set_ref, 0, 0)
        ab_grid.addWidget(self.btn_clear_ref, 0, 1)
        ab_grid.addWidget(self.label_ref_status, 1, 0, 1, 2)
        ab_grid.addWidget(self.label_drift_raw, 2, 0, 1, 2)
        ab_grid.addWidget(self.label_drift_proc, 3, 0, 1, 2)
        ab_grid.addWidget(self.label_stats_win, 4, 0, 1, 2)
        ab_grid.addWidget(self.label_cep_raw, 5, 0, 1, 2)
        ab_grid.addWidget(self.label_cep_proc, 6, 0, 1, 2)
        ab_grid.addWidget(self.label_improve_cep95, 7, 0, 1, 2)
        ab_box.setLayout(ab_grid)
        right_layout.addWidget(ab_box)

        # (3) 效能圖
        perf_box = QGroupBox("效能評估（最近 30 秒）")
        vbox_perf = QVBoxLayout()
        self.fig = Figure(figsize=(4,3))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("時間 (s)")
        self.ax.set_ylabel("值 / 毫秒")
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        self.lines = {
            "gps": self.ax.plot([], [], label="GPS Hz")[0],
            "imu": self.ax.plot([], [], label="IMU Hz")[0],
            "sys": self.ax.plot([], [], label="System Hz")[0],
            "infer": self.ax.plot([], [], label="Inference ms")[0],
        }
        self.ax.legend(loc="upper right", fontsize=8)
        vbox_perf.addWidget(self.canvas)
        perf_box.setLayout(vbox_perf)
        right_layout.addWidget(perf_box)

        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    # ------------------------- 參考點控制 -------------------------
    def on_set_reference(self):
        if hasattr(self, 'last_kf_xy') and self.last_kf_xy is not None:
            self.ref_xy = np.array(self.last_kf_xy, dtype=float)
            self.label_ref_status.setText("參考狀態：已手動設定 (使用 KF 當下位置)")
            self.label_ref_status.setStyleSheet("color:#1a7f37;")
        else:
            self.label_ref_status.setText("參考狀態：設定失敗（KF 尚未更新）")
            self.label_ref_status.setStyleSheet("color:#b3261e;")

    def on_clear_reference(self):
        self.ref_xy = None
        self.label_ref_status.setText(f"參考狀態：未設定（自動使用最近 {AUTO_REF_WINDOW_SEC}s KF 中位數）")
        self.label_ref_status.setStyleSheet("color:#444;")

    # ------------------------- 執行緒啟動 -------------------------
    def start_serial_threads(self):
        imu_thread = IMUSerialReader(IMU_PORT, self.shared_data)
        imu_thread.start()
        self.threads = [imu_thread]
        for port in COM_PORTS:
            t = SerialReader(port, self.shared_data, self.timestamps)
            t.start()
            self.threads.append(t)

    # ------------------------- 地圖載入 -------------------------
    def on_map_loaded(self, ok):
        if not ok:
            print("[ERROR] map.html load failed")
            return
        print("[INFO] map.html loaded successfully. Starting update timer.")
        self.timer.start(UPDATE_INTERVAL_MS)

    def safe_update_markers(self, js_code):
        def _cb(exists):
            try:
                if exists:
                    self.map_view.page().runJavaScript(js_code)
                else:
                    print("[WARN] updateMarkers is not defined yet in the page.")
            except Exception as e:
                print("[JS RUN ERROR]", e)
        self.map_view.page().runJavaScript("typeof updateMarkers !== 'undefined';", _cb)

    # ------------------------- 特徵提取（51 維） -------------------------
    def extract_features(self, lats, lons, rs, sats):
        origin_lat = np.mean(lats)
        xy = [latlon_to_xy(lat, lon, origin_lat) for lat, lon in zip(lats, lons)]
        d12 = np.linalg.norm(np.array(xy[0]) - np.array(xy[1]))
        d13 = np.linalg.norm(np.array(xy[0]) - np.array(xy[2]))
        d23 = np.linalg.norm(np.array(xy[1]) - np.array(xy[2]))
        vec_AB = np.array(xy[1]) - np.array(xy[0])
        vec_BC = np.array(xy[2]) - np.array(xy[1])
        vec_CA = np.array(xy[0]) - np.array(xy[2])
        dist_sum = d12**2 + d13**2 + d23**2
        r_mean = float(np.mean(rs))
        r_std  = float(np.std(rs))
        base_features = np.array([
            sats[0], sats[1], sats[2],
            rs[0], rs[1], rs[2],
            d12, d13, d23,
            rs[0]-rs[1], rs[0]-rs[2], rs[1]-rs[2],
            (rs[0]/rs[1]) if rs[1]!=0 else 0.0,
            (rs[0]/rs[2]) if rs[2]!=0 else 0.0,
            (rs[1]/rs[2]) if rs[2]!=0 else 0.0,
            vec_AB[0], vec_AB[1], vec_BC[0], vec_BC[1], vec_CA[0], vec_CA[1],
            dist_sum, r_mean, r_std
        ], dtype=np.float32)
        imu_features = []
        if 'IMU' in self.shared_data:
            history = self.shared_data['IMU']
            for i in range(9):
                hist = history.get(i, [])
                if len(hist) > 0:
                    mean_val = float(np.mean(hist))
                    std_val = float(np.std(hist))
                    delta_val = float(hist[-1] - hist[0])
                else:
                    mean_val = std_val = delta_val = 0.0
                imu_features.extend([mean_val, std_val, delta_val])
        else:
            imu_features = [0.0] * 27
        feat = np.concatenate([base_features, np.array(imu_features, dtype=np.float32)])
        return feat

    # ------------------------- 效能圖每秒更新 -------------------------
    def update_performance_plot(self):
        perf = self.shared_data.setdefault('_perf', {})
        gps_hz = perf.pop('gps', 0)
        imu_hz = perf.pop('imu', 0)
        sys_hz = self.sys_count
        self.sys_count = 0
        infer_ms = float(np.mean(self.infer_times_bucket)) if self.infer_times_bucket else 0.0
        self.infer_times_bucket = []

        t = int(time.time() - self.perf_time0)
        self.perf_history["time"].append(t)
        self.perf_history["gps"].append(gps_hz)
        self.perf_history["imu"].append(imu_hz)
        self.perf_history["sys"].append(sys_hz)
        self.perf_history["infer"].append(infer_ms)

        # 更新折線圖（最近 30 秒）
        xs = list(self.perf_history["time"])[-30:]
        for key in ("gps","imu","sys","infer"):
            ys = list(self.perf_history[key])[-30:]
            self.lines[key].set_data(xs, ys)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # 快照，供 CSV logging 使用
        self.last_perf_snapshot = {"gps": gps_hz, "imu": imu_hz, "sys": sys_hz, "infer": infer_ms}

    # ------------------------- CEP 計算 -------------------------
    @staticmethod
    def cep_from_distances(dists):
        if len(dists) == 0:
            return None, None
        arr = np.sort(np.array(dists, dtype=float))
        def percentile(p):
            idx = int(np.ceil(p/100.0 * len(arr))) - 1
            idx = max(0, min(idx, len(arr)-1))
            return float(arr[idx])
        return percentile(50), percentile(95)

    # ------------------------- 主更新（5Hz） -------------------------
    def update_gps_data(self):
        # 三顆 GPS 都有資料才更新
        if not all(p in self.shared_data for p in COM_PORTS):
            return
        latest_time = max(self.timestamps.values()) if self.timestamps else None
        if latest_time is None:
            return
        if self.last_processed_time and latest_time <= self.last_processed_time:
            return
        self.last_processed_time = latest_time
        self.sys_count += 1

        gps_data = [self.shared_data[p] for p in COM_PORTS]
        lats, lons, rs, sats = zip(*gps_data)
        origin_lat = float(np.mean(lats))
        xy = [latlon_to_xy(lat, lon, origin_lat) for lat, lon in zip(lats, lons)]

        # 動態 R 下限歷史（避免權重暴衝）
        for port, r in zip(COM_PORTS, rs):
            self.r_history[port].append(float(r))
            if len(self.r_history[port]) > 50:
                self.r_history[port].pop(0)

        # Raw（不加權中心）
        raw_xy = np.mean(np.array(xy), axis=0)

        # 加權中心（for 幾何/融合）
        weights = []
        for r, sat, port in zip(rs, sats, COM_PORTS):
            hist = self.r_history.get(port, [])
            r_clip = min(hist) if len(hist) > 0 else float(r)
            r_eff = max(float(r), r_clip)
            r_w = 1.0 / r_eff if r_eff > 0 else 1.0
            sat_w = sat if sat > 0 else 1
            weights.append(r_w * sat_w)
        weights = np.array(weights, dtype=float)
        weights = weights / np.sum(weights)
        center_w = np.sum([w * np.array(p) for w, p in zip(weights, xy)], axis=0)

        # KF 初值
        if self.kf.x[0] == 0 and self.kf.x[1] == 0:
            self.kf.x[0], self.kf.x[1] = center_w[0], center_w[1]

        # ---- MLP 推論（輸出 Δx, Δy）----
        mlp_xy = None
        mlp_lat = mlp_lon = None
        try:
            features = self.extract_features(lats, lons, rs, sats)
            self.feature_buffer.append(features)
            if len(self.feature_buffer) < WINDOW:
                while len(self.feature_buffer) < WINDOW:
                    self.feature_buffer.append(features)
            block = np.array(self.feature_buffer).reshape(1, -1)
            if self.scaler is not None:
                block = self.scaler.transform(block)
            input_tensor = torch.tensor(block, dtype=torch.float32)

            t0 = time.time()
            self.model.eval()
            with torch.no_grad():
                dxdy = self.model(input_tensor).cpu().numpy()[0]
            t1 = time.time()
            self.infer_times_bucket.append((t1 - t0) * 1000.0)

            mlp_xy = raw_xy + dxdy
            mlp_lat, mlp_lon = xy_to_latlon(mlp_xy[0], mlp_xy[1], origin_lat)
        except Exception as e:
            print("[MLP ERROR]", e)
            mlp_xy = None

        # ---- 幾何最小平方 ----
        def residuals(p):
            return [np.linalg.norm(p - np.array(xy[i])) - rs[i] for i in range(3)]
        try:
            geo_result = least_squares(residuals, center_w)
            geo_xy = geo_result.x
        except Exception as e:
            print("[GEO ERROR]", e)
            geo_xy = center_w
        geo_lat, geo_lon = xy_to_latlon(geo_xy[0], geo_xy[1], origin_lat)

        # ---- 融合（MLP + 幾何） ----
        if mlp_xy is not None:
            mlp_ws = []
            for r, sat, port in zip(rs, sats, COM_PORTS):
                hist = self.r_history.get(port, [])
                r_clip = min(hist) if len(hist) > 0 else float(r)
                r_eff = max(float(r), r_clip)
                w = (sat / (r_eff**2)) if r_eff > 0 else float(sat)
                mlp_ws.append(w)
            mlp_w = float(np.mean(mlp_ws))
            geo_w = 1.0
            fused_xy = (mlp_w * mlp_xy + geo_w * geo_xy) / (mlp_w + geo_w)
        else:
            fused_xy = geo_xy
        fused_lat, fused_lon = xy_to_latlon(fused_xy[0], fused_xy[1], origin_lat)

        # ---- KF 平滑（處理後） ----
        self.kf.predict(dt=UPDATE_INTERVAL_MS / 1000.0)
        kf_xy = self.kf.update(np.array(fused_xy))
        kf_lat, kf_lon = xy_to_latlon(kf_xy[0], kf_xy[1], origin_lat)
        self.last_kf_xy = kf_xy

        # 參考點（自動 / 手動）
        now = time.time()
        self.auto_ref_buffer.append((now, kf_xy[0], kf_xy[1]))
        while self.auto_ref_buffer and (now - self.auto_ref_buffer[0][0] > AUTO_REF_WINDOW_SEC):
            self.auto_ref_buffer.popleft()
        if self.ref_xy is not None:
            ref_xy = self.ref_xy
            self.label_ref_status.setText("參考狀態：手動（KF 當下位置）")
            self.label_ref_status.setStyleSheet("color:#1a7f37;")
        else:
            if len(self.auto_ref_buffer) >= 5:
                xs = np.array([p[1] for p in self.auto_ref_buffer])
                ys = np.array([p[2] for p in self.auto_ref_buffer])
                ref_xy = np.array([np.median(xs), np.median(ys)], dtype=float)
                self.label_ref_status.setText(f"參考狀態：自動（最近 {AUTO_REF_WINDOW_SEC}s KF 中位數）")
                self.label_ref_status.setStyleSheet("color:#444;")
            else:
                ref_xy = None
                self.label_ref_status.setText("參考狀態：等待建立自動參考...")
                self.label_ref_status.setStyleSheet("color:#888;")

        # UI：座標文字
        avg_lat, avg_lon = float(np.mean(lats)), float(np.mean(lons))
        self.label_gps.setText(f"原始GPS座標(紅色): {avg_lat:.6f}, {avg_lon:.6f}")
        self.label_geometry.setText(f"幾何計算座標(紫色): {geo_lat:.6f}, {geo_lon:.6f}")
        if mlp_lat is not None:
            self.label_mlp.setText(f"MLP預測座標(綠色): {mlp_lat:.6f}, {mlp_lon:.6f}")
        else:
            self.label_mlp.setText("MLP預測座標(綠色): 計算失敗")
        self.label_fused.setText(f"融合座標(藍色): {fused_lat:.6f}, {fused_lon:.6f}")
        self.label_kf.setText(f"Kalman Filter座標(橙色): {kf_lat:.6f}, {kf_lon:.6f}")
        self.label_sats.setText(f"各GPS衛星數: {sats[0]}, {sats[1]}, {sats[2]}")

        # Raw vs Processed（漂移與統計）
        def dist(a, b): return float(np.linalg.norm(np.array(a) - np.array(b)))
        processed_xy = kf_xy if PROCESSED_KIND == "KF" else fused_xy

        d_raw = None
        d_proc = None
        cep50_raw = cep95_raw = None
        cep50_proc = cep95_proc = None
        improve = None

        if ref_xy is not None:
            d_raw = dist(raw_xy, ref_xy)
            d_proc = dist(processed_xy, ref_xy)
            self.label_drift_raw.setText(f"Raw 即時漂移: {d_raw:.2f} m")
            self.label_drift_proc.setText(f"Processed 即時漂移: {d_proc:.2f} m")

            # 視窗緩衝
            self.stats_raw.append((now, d_raw))
            self.stats_proc.append((now, d_proc))
            while self.stats_raw and (now - self.stats_raw[0][0] > STATS_WINDOW_SEC):
                self.stats_raw.popleft()
            while self.stats_proc and (now - self.stats_proc[0][0] > STATS_WINDOW_SEC):
                self.stats_proc.popleft()

            # CEP
            cep50_raw, cep95_raw = self.cep_from_distances([v for _, v in self.stats_raw])
            cep50_proc, cep95_proc = self.cep_from_distances([v for _, v in self.stats_proc])
            if cep50_raw is not None:
                self.label_cep_raw.setText(f"Raw CEP50/CEP95: {cep50_raw:.2f} / {cep95_raw:.2f} m")
            else:
                self.label_cep_raw.setText("Raw CEP50/CEP95: - / - m")
            if cep50_proc is not None:
                self.label_cep_proc.setText(f"Processed CEP50/CEP95: {cep50_proc:.2f} / {cep95_proc:.2f} m")
            else:
                self.label_cep_proc.setText("Processed CEP50/CEP95: - / - m")

            # 改善（CEP95）
            if (cep95_raw is not None) and (cep95_proc is not None) and (cep95_raw > 0):
                improve = (cep95_raw - cep95_proc) / cep95_raw * 100.0
                self.label_improve_cep95.setText(f"改善（CEP95）：{improve:+.1f}%")
                self.label_improve_cep95.setStyleSheet("color:#1a7f37;" if improve >= 0 else "color:#b3261e;")
            else:
                self.label_improve_cep95.setText("改善（CEP95）：-")
                self.label_improve_cep95.setStyleSheet("color:#444;")
        else:
            self.label_drift_raw.setText("Raw 即時漂移: -")
            self.label_drift_proc.setText("Processed 即時漂移: -")
            self.label_cep_raw.setText("Raw CEP50/CEP95: - / - m")
            self.label_cep_proc.setText("Processed CEP50/CEP95: - / - m")
            self.label_improve_cep95.setText("改善（CEP95）：-")
            self.label_improve_cep95.setStyleSheet("color:#444;")

        # 地圖標記更新（map.html 需有 updateMarkers）
        js_code = (
            "updateMarkers({"
            f"gps:[{avg_lat},{avg_lon}],"
            f"geom:[{geo_lat},{geo_lon}],"
            f"mlp:[{mlp_lat if mlp_lat is not None else 'null'},{mlp_lon if mlp_lon is not None else 'null'}],"
            f"fused:[{fused_lat},{fused_lon}],"
            f"kf:[{kf_lat},{kf_lon}]"
            "});"
        )
        self.safe_update_markers(js_code)

        # ---- CSV 紀錄 ----
        # 取效能快照（上一秒統計值）
        gps_hz = self.last_perf_snapshot.get("gps")
        imu_hz = self.last_perf_snapshot.get("imu")
        sys_hz = self.last_perf_snapshot.get("sys")
        infer_ms = self.last_perf_snapshot.get("infer")
        # 若為 None，記 0
        gps_hz = 0 if gps_hz is None else gps_hz
        imu_hz = 0 if imu_hz is None else imu_hz
        sys_hz = 0 if sys_hz is None else sys_hz
        infer_ms = 0.0 if infer_ms is None else float(infer_ms)

        ts = datetime.now().isoformat(timespec='seconds')
        row = [ts,
               f"{d_raw:.3f}" if d_raw is not None else "",
               f"{d_proc:.3f}" if d_proc is not None else "",
               f"{cep50_raw:.3f}" if cep50_raw is not None else "",
               f"{cep95_raw:.3f}" if cep95_raw is not None else "",
               f"{cep50_proc:.3f}" if cep50_proc is not None else "",
               f"{cep95_proc:.3f}" if cep95_proc is not None else "",
               f"{improve:.2f}" if improve is not None else "",
               gps_hz, imu_hz, sys_hz, f"{infer_ms:.2f}"]
        try:
            with open(CSV_FILE, "a", newline="") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print("[CSV WRITE ERROR]", e)

# ===== 主程式 =====
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GPSViewer()
    win.show()
    exitcode = app.exec_()

    # 結束後畫圖（若有 ab_eval.csv）
# 結束後畫圖（若有 ab_eval.csv）
    if os.path.exists(CSV_FILE):
        try:
            import pandas as pd
            df = pd.read_csv(CSV_FILE)

            # ➡ 換成相對秒數
            t = np.arange(len(df)) * (UPDATE_INTERVAL_MS / 1000.0)
            df["CEP95_Raw_smooth"] = df["CEP95_Raw"].rolling(window=10, min_periods=1).mean()
            df["CEP95_Proc_smooth"] = df["CEP95_Proc"].rolling(window=10, min_periods=1).mean()
            df["Improve_CEP95_smooth"] = df["Improve_CEP95"].rolling(window=10, min_periods=1).mean()
            
            # 漂移曲線
            plt.figure(figsize=(11,6))
            if "Raw_Drift" in df.columns:
                plt.plot(t, df["Raw_Drift"], label="Raw Drift")
            if "Proc_Drift" in df.columns:
                plt.plot(t, df["Proc_Drift"], label="Processed Drift")
            plt.xlabel("Time (s)")
            plt.ylabel("Drift (m)")
            plt.title("Raw vs Processed Drift")
            plt.legend()
            plt.tight_layout()
            plt.savefig("drift_curve.png", dpi=300)   # 存檔
            plt.show()

            # CEP95 曲線
            plt.figure(figsize=(11,6))
            if "CEP95_Raw" in df.columns:
                plt.plot(t, df["CEP95_Raw_smooth"], label="CEP95 Raw")
            if "CEP95_Proc" in df.columns:
                plt.plot(t, df["CEP95_Proc_smooth"], label="CEP95 Proc")
            plt.xlabel("Time (s)")
            plt.ylabel("CEP95 (m)")
            plt.title("Raw vs Processed CEP95")
            plt.legend()
            plt.tight_layout()
            plt.savefig("cep95_curve.png", dpi=300)   # 存檔
            plt.show()

            # 改善百分比曲線
            plt.figure(figsize=(11,5))
            if "Improve_CEP95" in df.columns:
                plt.plot(t, df["Improve_CEP95_smooth"], label="Improve (CEP95 %)")
            plt.xlabel("Time (s)")
            plt.ylabel("Improve (%)")
            plt.title("CEP95 Improvement")
            plt.legend()
            plt.tight_layout()
            plt.savefig("improve_curve.png", dpi=300)   # 存檔
            plt.show()

            # ===== 整體平均改善率 =====
            if "CEP95_Raw" in df.columns and "CEP95_Proc" in df.columns:
                mean_raw = df["CEP95_Raw"].dropna().astype(float).mean()
                mean_proc = df["CEP95_Proc"].dropna().astype(float).mean()
                if mean_raw > 0:
                    overall_improve = (mean_raw - mean_proc) / mean_raw * 100
                    print("\n=== 整體平均結果 ===")
                    print(f"Raw 平均 CEP95: {mean_raw:.2f} m")
                    print(f"Processed 平均 CEP95: {mean_proc:.2f} m")
                    print(f"改善率: {overall_improve:+.1f} %")

        except Exception as e2:
                print("[PLOT FALLBACK ERROR]", e2)
