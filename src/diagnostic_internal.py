import mne
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import *
from master_connectivity import rid_rihaczek_gpu

def diagnostic_connectivity_phase(sub_id='sub-006'):
    print(f"🔍 [DIAGNOSTIC 1] Kiểm tra tính nhất quán của Pha (RID-Rihaczek) cho {sub_id}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load epochs
    epo_file = EPOCHS_DIR / f"{sub_id}_master-epo.fif"
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)
    
    # Chọn kênh FCz (nơi ERN mạnh nhất)
    ch_idx = epochs.ch_names.index('FCz') if 'FCz' in epochs.ch_names else 0
    data_raw = epochs['Incorrect'].get_data()[:, ch_idx, :256] * 1e6
    data_tensor = torch.tensor(data_raw, dtype=torch.float32, device=device)
    
    # Trích xuất phức số từ RID
    C = rid_rihaczek_gpu(data_tensor) # (trials, time, freq)
    
    # Lấy trung bình trong dải Theta
    freq_axis = np.fft.fftfreq(256, 1/SFREQ)
    theta_mask = (freq_axis >= THETA_BAND[0]) & (freq_axis <= THETA_BAND[1])
    theta_complex = torch.mean(C[:, :, theta_mask], dim=2)
    
    # Tính ITC (Inter-Trial Coherence)
    # ITC = |mean(exp(j * phase))|
    phase_vectors = theta_complex / (torch.abs(theta_complex) + 1e-12)
    itc = torch.abs(torch.mean(phase_vectors, dim=0)).cpu().numpy()
    
    plt.figure(figsize=(10, 4))
    time_ms = np.linspace(-1000, 1000, 256)
    plt.plot(time_ms, itc)
    plt.axvline(0, color='red', linestyle='--')
    plt.title(f"ITC at FCz (Phase Consistency) - {sub_id}")
    plt.ylabel("ITC Value")
    plt.grid(True)
    plt.savefig(OUTPUTS_DIR / "diag_itc_check.png")
    
    print(f"  -> Max ITC sau Response: {np.max(itc[128:160]):.4f}")
    if np.max(itc[128:160]) < 0.2:
        print("  ❌ CẢNH BÁO: ITC quá thấp. Pha từ RID-Rihaczek có thể đang bị nhiễu!")
    else:
        print("  ✅ Pha có tính nhất quán ổn định.")

def diagnostic_subspace_tracking():
    print(f"\n🔍 [DIAGNOSTIC 2] Kiểm tra tỷ lệ lỗi Residual trong HO-RLSL...")
    tensor_inc = np.load(TENSOR_INCORRECT_FILE)
    n_subs, n_nodes, _, n_times = tensor_inc.shape
    
    # Giả lập 1 bước update để xem Residual
    # Lấy 1 lát cắt ngẫu nhiên
    X_t = torch.tensor(tensor_inc[0, :, :, 135], dtype=torch.float32)
    
    # HOSVD đơn giản để tìm Subspace hiện tại
    U, S, V = torch.svd(X_t)
    r = 5
    U_r = U[:, :r]
    
    # Năng lượng được giữ lại (Projected)
    X_projected = U_r @ U_r.T @ X_t @ U_r @ U_r.T
    energy_in = torch.norm(X_projected)**2
    energy_total = torch.norm(X_t)**2
    
    residual_ratio = (energy_total - energy_in) / energy_total
    print(f"  -> Residual Energy Ratio (t=peak): {residual_ratio:.4f}")
    
    if residual_ratio > 0.3:
        print("  ⚠️ CẢNH BÁO: Rank r=5 đang bỏ sót quá nhiều năng lượng (>30%).")
    else:
        print("  ✅ Rank r=5 đủ để bao quát mạng lưới.")

if __name__ == "__main__":
    diagnostic_connectivity_phase()
    diagnostic_subspace_tracking()
