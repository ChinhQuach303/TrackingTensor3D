import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

def visualize_connectivity_dynamics():
    print("🎨 Đang khởi tạo các biểu đồ phân tích động học mạng lưới...")
    
    # 1. Load Tensors
    tensor_corr = np.load(TENSOR_CORRECT_FILE) # (S, N, N, T)
    tensor_inc = np.load(TENSOR_INCORRECT_FILE)
    
    n_subs, n_nodes, _, n_times = tensor_corr.shape
    time_ms = np.linspace(-1000, 1000, n_times)
    
    # ---------------------------------------------------------
    # PLOT 1: Global Field Synchrony (Mean PLV over time)
    # ---------------------------------------------------------
    # Trung bình qua các cặp node (N,N) và qua subjects (S)
    mean_conn_corr = np.mean(tensor_corr, axis=(0, 1, 2))
    mean_conn_inc = np.mean(tensor_inc, axis=(0, 1, 2))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_ms, mean_conn_corr, label='Correct', color='blue', alpha=0.8)
    plt.plot(time_ms, mean_conn_inc, label='Incorrect (Error)', color='red', linewidth=2)
    plt.axvline(x=0, color='black', linestyle='--', label='Response')
    plt.fill_between(time_ms, mean_conn_corr, mean_conn_inc, where=(mean_conn_inc > mean_conn_corr), 
                     color='red', alpha=0.1, label='Error Excess')
    
    plt.title("Diễn biến kết nối tổng thể (Global Connectivity Dynamics)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Average PLV")
    plt.xlim(-200, 600) # Tập trung vào khoảng sau response
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUTS_DIR / "connectivity_dynamics_global.png")
    
    # ---------------------------------------------------------
    # PLOT 2: Degree Centrality Topomaps (The "Hubs")
    # ---------------------------------------------------------
    # Tính tại peak ERN (khoảng 50-150ms -> index 134-147)
    peak_win = slice(134, 147)
    degree_corr = np.mean(tensor_corr[:, :, :, peak_win], axis=(0, 2, 3))
    degree_inc = np.mean(tensor_inc[:, :, :, peak_win], axis=(0, 2, 3))
    
    # Load info để vẽ topomap
    raw_sample = mne.io.read_raw_eeglab(list(DATA_RAW.glob("sub-001/eeg/*.set"))[0], preload=False)
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw_sample.rename_channels(lambda x: rename_map.get(x, x))
    raw_sample.set_montage(MONTAGE_NAME, on_missing='ignore')
    
    # Lọc lại info cho 30 kênh EEG
    available_eeg = [ch for ch in EEG_CHANNELS if ch in raw_sample.ch_names]
    info = raw_sample.copy().pick_channels(available_eeg).info
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    mne.viz.plot_topomap(degree_corr, info, axes=axes[0], show=False, cmap='viridis')
    axes[0].set_title("Hubs: Correct (50-150ms)")
    
    mne.viz.plot_topomap(degree_inc, info, axes=axes[1], show=False, cmap='viridis')
    axes[1].set_title("Hubs: Incorrect (Error) (50-150ms)")
    
    plt.savefig(OUTPUTS_DIR / "connectivity_hubs_topomap.png")

    # ---------------------------------------------------------
    # PLOT 3: Difference Network (Incorrect - Correct)
    # ---------------------------------------------------------
    diff_matrix = np.mean(tensor_inc[:, :, :, peak_win], axis=(0, 3)) - \
                  np.mean(tensor_corr[:, :, :, peak_win], axis=(0, 3))
    
    plt.figure(figsize=(8, 7))
    plt.imshow(diff_matrix, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    plt.colorbar(label="$\Delta$ PLV")
    plt.title("Mạng lưới khác biệt (Incorrect - Correct) tại Peak ERN")
    plt.xlabel("Nodes")
    plt.ylabel("Nodes")
    plt.savefig(OUTPUTS_DIR / "connectivity_difference_matrix.png")

    print(f"✅ Đã lưu 3 biểu đồ phân tích tại: {OUTPUTS_DIR}")

if __name__ == "__main__":
    visualize_connectivity_dynamics()
