# ========== GPS + IMU 每0.2秒資料收集整合程式 ==========
# 本程式整合三顆 GPS 接收器與 IMU901 的數據，每0.2秒同步輸出一筆平均值至 CSV 檔案。
# ✅ 新增：GPS 解析資料包含 Date, Time, Lat, Lon, Alt, Spd, Crs, SatUsed, HDOP, NSatView, SNR 共 11 欄

import serial
import threading
import time
import csv
from datetime import datetime
import os
import struct

# ----------- 基本設定 -----------
baud_rate = 115200
output_csv = "gps_imu_data.csv"
CSV_FILE = output_csv  # 統一使用同一個 CSV 檔案

# ----------- 從設定檔讀取 COM 對應位置名稱 -----------
def load_com_map(config_file="com_mapping_config.txt"):
    com_map = {}
    try:
        with open(config_file, 'r') as f:
            for line in f:
                if ':' in line:
                    com, role = line.strip().split(':')
                    com_map[com] = role
        if len(com_map) != 3:
            raise ValueError("❌ COM 對應不足三個，請重新設定")
    except Exception as e:
        print(f"❌ 無法讀取 COM 對應檔案: {e}")
        exit(1)
    return com_map

COM_MAP = load_com_map()
arduino_COM = list(COM_MAP.keys())

# ----------- 資料儲存結構與旗標 -----------
COM_GPS_data = {port: None for port in arduino_COM}
stop_flag = False
data_lock = threading.Lock()
count = 0

# ----------- GPS 資料讀取執行緒 -----------
def read_gps(port):
    global COM_GPS_data
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        while not stop_flag:
            if ser.in_waiting:
                data = ser.readline().decode(errors='ignore').strip()
                if 'ID' not in data:
                    try:
                        # ✅ 新格式: Date,Time,Lat,Lon,Alt,Spd,Crs,SatUsed,HDOP,NSatView
                        parts = data.split(',')
                        if len(parts) >= 10:
                            date_str = parts[0]
                            time_str = parts[1]
                            lat = float(parts[2])
                            lon = float(parts[3])
                            alt = float(parts[4])
                            spd = float(parts[5])
                            crs = float(parts[6])
                            sats_used = int(parts[7])
                            hdop = float(parts[8])
                            nsat_view = int(parts[9])
                            with data_lock:
                                COM_GPS_data[port] = [
                                    date_str, time_str, lat, lon, alt,
                                    spd, crs, sats_used, hdop,
                                    nsat_view
                                ]
                    except ValueError:
                        print(f"[{port}] 無法解析資料：{data}")
    except Exception as e:
        print(f"[{port}] 錯誤: {e}")

# ---------- IMU 接收與平均程式碼 ----------
def read_imu():
    imu_ser = serial.Serial('COM11', 115200, timeout=0.5)
    accum_acc = [0.0, 0.0, 0.0]
    accum_gyro = [0.0, 0.0, 0.0]
    accum_count = 0
    roll = pitch = yaw = None
    last_time = time.time()

    while not stop_flag:
        if imu_ser.in_waiting >= 9:
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

        # ✅ 每0.2秒輸出一次平均值
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

# ----------- CSV 初始化 -----------
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'A_Date', 'A_Time', 'A_Lat', 'A_Lon', 'A_Alt', 'A_Spd', 'A_Crs', 'A_SatUsed', 'A_HDOP', 'A_NSatView',
            'B_Date', 'B_Time', 'B_Lat', 'B_Lon', 'B_Alt', 'B_Spd', 'B_Crs', 'B_SatUsed', 'B_HDOP', 'B_NSatView', 
            'C_Date', 'C_Time', 'C_Lat', 'C_Lon', 'C_Alt', 'C_Spd', 'C_Crs', 'C_SatUsed', 'C_HDOP', 'C_NSatView', 
            'Roll', 'Pitch', 'Yaw',
            'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz',
            'Timestamp'
        ])

# ----------- 啟動所有執行緒 -----------
imu_data_queue = []
threading.Thread(target=read_imu, daemon=True).start()
for port in arduino_COM:
    threading.Thread(target=read_gps, args=(port,), daemon=True).start()

# ----------- 主流程，每0.2秒收集同步資料並寫入 -----------
try:
    while count < 10000:
        time.sleep(0.2)
        with data_lock:
            if all(COM_GPS_data[port] is not None for port in arduino_COM) and imu_data_queue:
                gps_order = ['A', 'B', 'C']
                ordered_data = [
                    COM_GPS_data[port] 
                    for port, role in sorted(COM_MAP.items(), key=lambda x: gps_order.index(x[1]))
                ]
                imu_values = imu_data_queue.pop(0)

                print(f"收集到 GPS 資料:")
                for label, data in zip(gps_order, ordered_data):
                    print(f"{label}: {data}")
                print(f"IMU: {imu_values}")

                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = []
                    for data in ordered_data:
                        row.extend(data)
                    row.extend(imu_values)
                    row.append(datetime.now().isoformat())
                    writer.writerow(row)

                count += 1
                print(f"✅ 已寫入第 {count} 筆資料")

                for port in arduino_COM:
                    COM_GPS_data[port] = None

    stop_flag = True
    print("✅ 已收集滿資料，自動結束程式")

except KeyboardInterrupt:
    stop_flag = True
    print("🛑 結束資料收集")
