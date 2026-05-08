import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from config import *

def analyze_network_states():
    print("🧠 Đang phân tích mối liên hệ giữa ERP và Trạng thái mạng lưới...")
    
    # 1. Load dữ liệu Energy và Change-points
    energy_inc = np.load(TENSOR_DIR / "horls_energy_incorrect.npy")
    energy_cor = np.load(TENSOR_DIR / "horls_energy_correct.npy")
    time_ms = np.linspace(-1000, 1000, 256)
    
    # Hàm tìm Change-points (tái sử dụng logic từ tensor_decomposition)
    def get_cp(energy):
        diff_energy = np.abs(np.diff(energy))
        threshold = np.mean(diff_energy) + 1.5 * np.std(diff_energy)
        indices = np.where(diff_energy > threshold)[0]
        if len(indices) == 0: return []
        distinct = [indices[0]]
        for idx in indices[1:]:
            if idx - distinct[-1] > 20: distinct.append(idx)
        return distinct

    cp_inc = get_cp(energy_inc)
    cp_cor = get_cp(energy_cor)

    # 2. Tính Grand Average ERP (tại FCz)
    sub_files = list(EPOCHS_DIR.glob("*_master-epo.fif"))
    all_erp_inc = []
    all_erp_cor = []
    
    for f in sub_files:
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        ch_idx = epochs.ch_names.index('FCz') if 'FCz' in epochs.ch_names else 0
        all_erp_inc.append(np.mean(epochs['Incorrect'].get_data()[:, ch_idx, :256], axis=0))
        all_erp_cor.append(np.mean(epochs['Correct'].get_data()[:, ch_idx, :256], axis=0))
        
    ga_erp_inc = np.mean(all_erp_inc, axis=0) * 1e6 # μV
    ga_erp_cor = np.mean(all_erp_cor, axis=0) * 1e6
    
    # --- BIỂU ĐỒ 1: ĐỐI CHIẾU ERP VÀ NETWORK ENERGY ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # Vẽ ERP
    ax1.plot(time_ms, ga_erp_cor, label='Correct ERP (FCz)', color='blue', alpha=0.6)
    ax1.plot(time_ms, ga_erp_inc, label='Incorrect ERP (FCz)', color='red', linewidth=2)
    ax1.set_title("Classical ERP Components (ERN/Pe/RP)", fontsize=14)
    ax1.set_ylabel("Amplitude (μV)")
    ax1.axvline(0, color='black', linestyle='-')
    ax1.legend()
    
    # Chú thích các đỉnh sóng
    ax1.annotate('Readiness Potential', xy=(-420, ga_erp_inc[int(256*0.58/2)]), xytext=(-600, 5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax1.annotate('ERN', xy=(50, ga_erp_inc[int(256*1.05/2)]), xytext=(100, -10),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Vẽ Network Energy & Change-points
    ax2.plot(time_ms, energy_cor, color='blue', alpha=0.5, label='Correct Energy')
    ax2.plot(time_ms, energy_inc, color='red', linewidth=2, label='Incorrect Energy')
    for cp in cp_inc:
        ax2.axvline(time_ms[cp], color='red', linestyle='--', alpha=0.5)
    ax2.set_title("Network Energy Dynamics & Change-points", fontsize=14)
    ax2.set_ylabel("Energy")
    ax2.set_xlabel("Time (ms)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "erp_network_correlation.png")
    
    # --- BIỂU ĐỒ 2: STATE DURATION ANALYSIS ---
    def calc_durations(cp_indices):
        # Giả định trạng thái bắt đầu từ 0 và kết thúc ở 255
        full_indices = [0] + sorted(cp_indices) + [255]
        intervals = np.diff(full_indices) * (2000 / 256) # Chuyển sang ms
        return intervals

    dur_inc = calc_durations(cp_inc)
    dur_cor = calc_durations(cp_cor)
    
    plt.figure(figsize=(10, 6))
    labels = ['Incorrect', 'Correct']
    means = [np.mean(dur_inc), np.mean(dur_cor)]
    stds = [np.std(dur_inc), np.std(dur_cor)]
    
    plt.bar(labels, means, yerr=stds, color=['red', 'blue'], alpha=0.6, capsize=10)
    plt.title("Mean Network State Duration (Stability Analysis)", fontsize=14)
    plt.ylabel("Mean Duration (ms)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(OUTPUTS_DIR / "network_state_stability.png")
    
    print(f"📊 Đã hoàn tất phân tích. Kết quả lưu tại: {OUTPUTS_DIR}")
    print(f"  -> Mean Duration (Incorrect): {means[0]:.2f} ms")
    print(f"  -> Mean Duration (Correct): {means[1]:.2f} ms")

if __name__ == "__main__":
    analyze_network_states()
