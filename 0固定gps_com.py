# ✅ 本程式用於列出目前可用的 COM port 並讓使用者手動指定 A/B/C 三個 GPS 對應的序號
# 設定完會把結果寫入 `com_mapping_config.txt` 給主程式讀取使用。
# -----------------------------------------------------------------------


import serial.tools.list_ports

def list_available_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]

def main():
    print("🔌 可用的 COM port：")
    available_ports = list_available_ports()
    for i, port in enumerate(available_ports):
        print(f"[{i}] {port}")

    if len(available_ports) < 3:
        print("❌ 需要至少 3 個 COM port")
        return

    print("\n請輸入各點對應的 COM 編號：")
    A_idx = int(input("請選擇 A 點的 COM 編號："))
    B_idx = int(input("請選擇 B 點的 COM 編號："))
    C_idx = int(input("請選擇 C 點的 COM 編號："))

    selected = {
        available_ports[A_idx]: 'A',
        available_ports[B_idx]: 'B',
        available_ports[C_idx]: 'C'
    }

    print("\n✅ 選擇結果：")
    for com, role in selected.items():
        print(f"{role}: {com}")

    with open("com_mapping_config.txt", "w") as f:
        for com, role in selected.items():
            f.write(f"{com}:{role}\n")

    print("\n📁 已儲存為 com_mapping_config.txt，可於主程式讀取使用")

if __name__ == "__main__":
    main()
