# ==========================
# GPS + IMU MLP 推論 + 測試評估（時間窗版）
# ==========================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from math import sqrt

# 時間窗長度（要和訓練程式一致）
WINDOW = 5

# -------------------------
# Dataset (加入時間窗，保留 center)
# -------------------------
class GPSPTDataset(Dataset):
    def __init__(self, pt_file, window=WINDOW, scaler_path="scaler.pkl"):
        data = torch.load(pt_file)
        X = data['features'].numpy()   # [N, 33]
        Y = data['target'].numpy()     # fused E/N
        C = data['center'].numpy()     # 幾何中心 E/N

        # ΔE/ΔN = fused - center
        Y = Y - C

        # 防止 NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        # 組合時間窗
        Xw_list, Yw_list, Cw_list = [], [], []
        for t in range(window - 1, len(X)):
            block = X[t - window + 1 : t + 1]  # (window, 33)
            Xw_list.append(block.reshape(-1))  # 攤平成 (33*window,)
            Yw_list.append(Y[t])               # 視窗最後一筆的 ΔE/ΔN
            Cw_list.append(C[t])               # 視窗最後一筆的 center

        Xw = np.stack(Xw_list, axis=0)
        Yw = np.stack(Yw_list, axis=0)
        Cw = np.stack(Cw_list, axis=0)

        # 使用訓練時的 scaler
        scaler = joblib.load(scaler_path)
        Xw = scaler.transform(Xw)

        self.X = torch.tensor(Xw, dtype=torch.float32)
        self.Y = torch.tensor(Yw, dtype=torch.float32)
        self.C = torch.tensor(Cw, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.C[idx]

# -------------------------
# 模型結構 (要和訓練一致)
# -------------------------
class GeoMLP(nn.Module):
    def __init__(self, input_dim=33 * WINDOW):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 輸出 ΔE, ΔN
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# 主程式
# -------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = GPSPTDataset('gps_features_all.pt', window=WINDOW, scaler_path="scaler.pkl")
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = GeoMLP(input_dim=dataset.X.shape[1]).to(device)
    model.load_state_dict(torch.load('best_gps_model.pth', map_location=device))
    model.eval()

    preds, gts, centers = [], [], []

    with torch.no_grad():
        for X_batch, Y_batch, C_batch in loader:
            X_batch, Y_batch, C_batch = X_batch.to(device), Y_batch.to(device), C_batch.to(device)
            pred = model(X_batch)

            preds.append(pred.cpu().numpy())
            gts.append(Y_batch.cpu().numpy())
            centers.append(C_batch.cpu().numpy())

    preds = np.vstack(preds)    # ΔE/ΔN 預測
    gts = np.vstack(gts)        # ΔE/ΔN 真值
    centers = np.vstack(centers)

    # -------------------------
    # 誤差統計
    # -------------------------
    errors = preds - gts
    mse = np.mean(errors**2)
    rmse = sqrt(mse)
    mae = np.mean(np.abs(errors))

    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f} m")
    print(f"Test MAE: {mae:.6f} m")

    # -------------------------
    # 還原最終座標 = center + ΔE/ΔN
    # -------------------------
    final_preds = centers + preds

    print("\n範例輸出 (前 5 筆):")
    for i in range(5):
        print(f"Center: {centers[i]}, Pred Δ: {preds[i]}, Final Pred: {final_preds[i]}")

if __name__ == "__main__":
    main()
