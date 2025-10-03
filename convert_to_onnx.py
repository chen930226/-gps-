import torch
import torch.nn as nn

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
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

# ----------------- 改這裡 -----------------
INPUT_DIM = 255   # 用訓練時的實際維度
# ------------------------------------------

model = GeoMLP(input_dim=INPUT_DIM)
model.load_state_dict(torch.load("best_gps_model.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, INPUT_DIM)

torch.onnx.export(
    model,
    dummy_input,
    "gps_mlp_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['delta_en'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'delta_en': {0: 'batch_size'}
    }
)

print("✅ 已成功匯出 ONNX 模型 → gps_mlp_model.onnx")
