# ==========================
# GPS + IMU MLP 訓練程式（33 維特徵 → 時間窗攤平，目標 = fused - center）
# ==========================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# 時間窗長度（可改），輸入維度將變成 33 * WINDOW
WINDOW = 5

# -------------------------
# Dataset 定義（加入時間窗攤平）
# -------------------------
class GPSPTDataset(Dataset):
    def __init__(self, pt_file, window=WINDOW):
        data = torch.load(pt_file)
        X = data['features'].numpy()   # [N, 33] (GPS+IMU 特徵)
        Y = data['target'].numpy()     # fused E/N
        C = data['center'].numpy()     # 幾何中心 E/N

        # 目標改成 ΔE, ΔN
        Y = Y - C

        # 防止 NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

        # 依時間建立滑動視窗
        Xw_list, Yw_list = [], []
        for t in range(window - 1, len(X)):
            block = X[t - window + 1 : t + 1]            # (window, 33)
            Xw_list.append(block.reshape(-1))            # (33*window,)
            Yw_list.append(Y[t])                         # 對應 ΔE/ΔN

        Xw = np.stack(Xw_list, axis=0)                   # [N', 33*window]
        Yw = np.stack(Yw_list, axis=0)                   # [N', 2]

        # 標準化 X
        scaler = StandardScaler()
        Xw = scaler.fit_transform(Xw)
        joblib.dump(scaler, "scaler.pkl")

        self.X = torch.tensor(Xw, dtype=torch.float32)
        self.Y = torch.tensor(Yw, dtype=torch.float32)
        self.window = window

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------------
# MLP 模型
# -------------------------
class GeoMLP(nn.Module):
    def __init__(self, input_dim):
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
# 訓練主程式
# -------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = GPSPTDataset('gps_features_all.pt', window=WINDOW)
    input_dim = dataset.X.shape[1]  # 33 * WINDOW
    print(f"Window = {WINDOW} → input_dim = {input_dim}")

    train_size = int(len(dataset) * 0.7)
    val_size   = int(len(dataset) * 0.15)
    test_size  = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = GeoMLP(input_dim=input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 20
    counter = 0
    num_epochs = 200

    train_losses_hist, val_losses_hist = [], []

    for epoch in range(1, num_epochs + 1):
        # -------------------------
        # 訓練
        # -------------------------
        model.train()
        train_losses = []
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # -------------------------
        # 驗證
        # -------------------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, Y_batch)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        train_losses_hist.append(avg_train_loss)
        val_losses_hist.append(avg_val_loss)

        print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_gps_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # -------------------------
    # 畫圖並儲存
    # -------------------------
    plt.figure(figsize=(10,5))
    plt.plot(train_losses_hist, label="Train Loss")
    plt.plot(val_losses_hist, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(f"MLP Training (GPS+IMU 33×{WINDOW} Features → ΔE/ΔN)")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    plt.close()

    print("✅ 訓練完成，最佳模型已儲存為 best_gps_model.pth，曲線已輸出 training_curve.png")

if __name__ == "__main__":
    main()
