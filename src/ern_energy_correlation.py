import mne
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from config import *
from tensor_decomposition import HORLSDecomposer

def analyze_ern_energy_correlation():
    print("📊 Đang kiểm tra tương quan giữa Biên độ ERN và Năng lượng mạng lưới...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load danh sách subjects đã xử lý thành công
    sub_files = sorted(list(EPOCHS_DIR.glob("*_master-epo.fif")))
    
    ern_amplitudes = []
    energy_bumps = []
    valid_subs = []

    # Khởi tạo decomposer để tính energy cho từng cá nhân (để lấy năng lượng riêng lẻ)
    # Lưu ý: Bài báo dùng energy chung, nhưng để tính correlation ta cần energy của từng sub
    
    for f in sub_files:
        sid = f.name.split('_')[0]
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        # Chỉ lấy điều kiện Incorrect
        ep_inc = epochs['Incorrect']
        if len(ep_inc) < MIN_INCORRECT_TRIALS: continue
        
        # A. Trích xuất biên độ ERN (Min value tại FCz trong 0-100ms)
        ch_idx = ep_inc.ch_names.index('FCz') if 'FCz' in ep_inc.ch_names else 0
        erp_inc = np.mean(ep_inc.get_data()[:, ch_idx, :], axis=0) * 1e6 # μV
        erp_inc = erp_inc[:256] # Đảm bảo khớp 256 mẫu
        
        time_ms = np.linspace(-1000, 1000, 256)
        ern_window = (time_ms >= 0) & (time_ms <= 100)
        ern_peak = np.min(erp_inc[ern_window]) # ERN là sóng âm nên lấy Min
        
        # B. Trích xuất "Bướu" năng lượng cho subject này
        # Chúng ta cần tính PLV riêng cho subject này để lấy energy riêng
        # (Bước này hơi tốn thời gian nhưng cần thiết cho Correlation)
        
        # Giả sử ta lấy năng lượng từ tensor đã lưu nếu có, hoặc tính nhanh tại đây
        # Để tối ưu, tôi sẽ lấy lát cắt năng lượng từ Tensor 4D đã lưu tại Giai đoạn 3
        # Nhưng energy trong decomposition là của Core Tensor (Common), 
        # nên ta sẽ xấp xỉ bằng cách tính Projected Energy của chính subject đó.
        
        ern_amplitudes.append(ern_peak)
        valid_subs.append(sid)

    # Đọc Tensor 4D đã lưu để tính energy riêng lẻ
    tensor_inc = np.load(TENSOR_INCORRECT_FILE) # (S, N, N, T)
    # Lấy các subject tương ứng
    # (Vì master_connectivity và decomposition lưu theo cùng thứ tự subject)
    
    for i in range(len(ern_amplitudes)):
        sub_tensor = tensor_inc[i, :, :, :] # (N, N, T)
        # Tính năng lượng mạng lưới đơn giản cho subject: norm của ma trận kết nối dải Theta
        sub_energy = np.linalg.norm(sub_tensor, axis=(0, 1))**2
        bump_val = np.max(sub_energy[ern_window])
        energy_bumps.append(bump_val)

    # 2. Tính toán tương quan
    r_val, p_val = pearsonr(ern_amplitudes, energy_bumps)
    
    # 3. Vẽ biểu đồ Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(ern_amplitudes, energy_bumps, color='red', alpha=0.6, s=100)
    
    # Vẽ đường hồi quy (Regression Line)
    m, b = np.polyfit(ern_amplitudes, energy_bumps, 1)
    plt.plot(ern_amplitudes, m*np.array(ern_amplitudes) + b, color='black', linestyle='--')
    
    plt.title(f"Correlation: ERN Amplitude vs. Network Energy Bump\n(r = {r_val:.3f}, p = {p_val:.4f})", fontsize=14)
    plt.xlabel("ERN Peak Amplitude (μV) - More negative is stronger")
    plt.ylabel("Network Energy Bump Magnitude")
    plt.grid(True, alpha=0.3)
    
    # Nghịch đảo trục X vì ERN càng âm càng mạnh
    plt.gca().invert_xaxis() 
    
    output_path = OUTPUTS_DIR / "ern_energy_correlation_scatter.png"
    plt.savefig(output_path)
    
    print(f"✅ Đã hoàn tất kiểm tra tương quan!")
    print(f"  -> Hệ số tương quan r: {r_val:.4f}")
    print(f"  -> Giá trị p-value: {p_val:.4f}")
    
    if p_val < 0.05:
        print("  🔥 KẾT QUẢ CÓ Ý NGHĨA THỐNG KÊ! ERN thực sự liên quan mật thiết đến năng lượng mạng lưới.")
    else:
        print("  ⚠️ Mối tương quan chưa rõ rệt ở mức p < 0.05.")

if __name__ == "__main__":
    analyze_ern_energy_correlation()
