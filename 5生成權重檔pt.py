# ===============================
# CSV → .pt 生成程式 (GPS 幾何 24 維 + IMU 統計特徵 27 維 = 51 維特徵)
# ===============================
import torch
import pandas as pd
import numpy as np
from math import radians, cos

EARTH_RADIUS = 6371000  # 地球半徑（公尺）

# 經緯度轉 XY
def latlon_to_xy(lat, lon, origin_lat):
    x = radians(lon) * EARTH_RADIUS * cos(radians(origin_lat))
    y = radians(lat) * EARTH_RADIUS
    return x, y

def weighted_fuse(lats, lons, rs, sats, origin_lat):
    weights = []
    for r, sat in zip(rs, sats):
        if r > 0:
            weights.append(max(sat,1.0) / max(r**2, 1e-6))
        else:
            weights.append(max(sat,1.0))
    wsum = sum(weights)
    lat_f = sum(w*l for w, l in zip(weights, lats)) / wsum
    lon_f = sum(w*l for w, l in zip(weights, lons)) / wsum
    return latlon_to_xy(lat_f, lon_f, origin_lat)

# ✅ 安全轉換函數
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

# -------------------------
# 主程式
# -------------------------
def main():
    # 讀取 CSV
    df = pd.read_csv("gps_imu_data2.csv")

    features_list = []
    fused_list = []
    centers_list = []

    # 🔹 每顆 GPS 的 R 歷史（最近 10 筆）— 用於動態下限
    r_histories = [[], [], []]  # 對應 A/B/C

    # 🔹 建立 IMU 歷史緩衝區
    imu_histories = {col: [] for col in ['Roll','Pitch','Yaw','Ax','Ay','Az','Gx','Gy','Gz']}
    IMU_WINDOW = 5

    for idx, row in df.iterrows():
        lats = [row['A_Lat'], row['B_Lat'], row['C_Lat']]
        lons = [row['A_Lon'], row['B_Lon'], row['C_Lon']]
        rs   = [row['A_R'], row['B_R'], row['C_R']]
        sats = [row['A_Sat'], row['B_Sat'], row['C_Sat']]

        origin_lat = np.mean(lats)
        xy = [np.array(latlon_to_xy(lat, lon, origin_lat)) for lat, lon in zip(lats, lons)]
        center = np.mean(xy, axis=0)

        # 幾何特徵
        d12 = np.linalg.norm(xy[0]-xy[1])
        d13 = np.linalg.norm(xy[0]-xy[2])
        d23 = np.linalg.norm(xy[1]-xy[2])
        vec_AB = xy[1]-xy[0]
        vec_BC = xy[2]-xy[1]
        vec_CA = xy[0]-xy[2]
        dist_sum = d12**2 + d13**2 + d23**2
        r_mean = np.mean(rs)
        r_std  = np.std(rs)

        base_features = np.array([
            sats[0], sats[1], sats[2],      # 衛星數 3
            rs[0], rs[1], rs[2],            # 半徑 3 → 6
            d12, d13, d23,                  # 距離 3 → 9
            rs[0]-rs[1], rs[0]-rs[2], rs[1]-rs[2],  # 3 → 12
            (rs[0]/rs[1]) if rs[1]!=0 else 0.0,     # 1 → 13
            (rs[0]/rs[2]) if rs[2]!=0 else 0.0,     # 1 → 14
            (rs[1]/rs[2]) if rs[2]!=0 else 0.0,     # 1 → 15
            vec_AB[0], vec_AB[1], vec_BC[0], vec_BC[1], vec_CA[0], vec_CA[1],  # 6 → 21
            dist_sum, r_mean, r_std         # 3 → 24
        ], dtype=np.float32)

        # 更新 IMU 歷史緩衝區
        for col in imu_histories.keys():
            imu_histories[col].append(safe_float(row[col]))
            if len(imu_histories[col]) > IMU_WINDOW:
                imu_histories[col].pop(0)

        # 計算 IMU 統計特徵（均值、標準差、變化量）
        imu_features = []
        for col in imu_histories.keys():
            hist = imu_histories[col]
            if len(hist) > 0:
                mean_val = np.mean(hist)
                std_val  = np.std(hist)
                delta_val = hist[-1] - hist[0]
            else:
                mean_val, std_val, delta_val = 0.0, 0.0, 0.0
            imu_features.extend([mean_val, std_val, delta_val])

        # 合併 base + IMU = 24 + 27 = 51 維
        feat = np.concatenate([base_features, np.array(imu_features, dtype=np.float32)])  # 51 維
        features_list.append(feat.astype(np.float32))

        # Target = fused E/N
        rs_eff = []
        for i in range(3):
            hist = r_histories[i]
            r_i = float(rs[i])
            r_clip = min(hist) if len(hist) > 0 else r_i
            r_eff = max(r_i, r_clip)
            rs_eff.append(r_eff)

        fused_xy = np.array(weighted_fuse(lats, lons, rs_eff, sats, origin_lat))

        # 更新 R 歷史
        for i in range(3):
            r_histories[i].append(float(rs[i]))
            if len(r_histories[i]) > 100:
                r_histories[i].pop(0)
        fused_list.append(fused_xy.astype(np.float32))

        # 同時存 center
        centers_list.append(center.astype(np.float32))

    # 轉成 tensor
    X = torch.tensor(np.stack(features_list), dtype=torch.float32)  # [N, 51]
    Y = torch.tensor(np.stack(fused_list), dtype=torch.float32)     # [N, 2]
    C = torch.tensor(np.stack(centers_list), dtype=torch.float32)   # [N, 2]

    # 儲存成 .pt
    torch.save({'features': X, 'target': Y, 'center': C}, 'gps_features_all.pt')
    print("✅ 已生成 gps_features_all.pt，維度:", X.shape, Y.shape, C.shape)

if __name__ == '__main__':
    main()
