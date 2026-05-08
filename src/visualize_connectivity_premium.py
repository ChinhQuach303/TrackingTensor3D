import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

def redraw_connectivity_plots():
    print("🎨 Đang vẽ lại các biểu đồ Connectivity Matrix (Premium Style)...")
    
    # 1. Load Balanced Tensors
    tensor_corr = np.load(TENSOR_CORRECT_FILE)
    tensor_inc = np.load(TENSOR_INCORRECT_FILE)
    
    # Lấy thông tin kênh từ một file epoch mẫu
    sample_epo = mne.read_epochs(list(EPOCHS_DIR.glob("*.fif"))[0], preload=False, verbose=False)
    ch_names = sample_epo.ch_names
    
    # 2. Tính trung bình tại Peak ERN (index ~135)
    peak_idx = 135
    adj_corr = np.mean(tensor_corr[:, :, :, peak_idx], axis=0)
    adj_inc = np.mean(tensor_inc[:, :, :, peak_idx], axis=0)
    adj_diff = adj_inc - adj_corr
    
    # --- BIỂU ĐỒ 1: COMPARISON (CORRECT vs INCORRECT) ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    for ax, data, title in zip(axes, [adj_corr, adj_inc], ['CORRECT Network', 'INCORRECT Network']):
        im = ax.imshow(data, cmap='magma', vmin=0, vmax=0.8)
        ax.set_title(f"{title} (Peak ERN)", fontsize=14, fontweight='bold')
        ax.set_xticks(np.arange(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(ch_names)))
        ax.set_yticklabels(ch_names, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='PLV')

    plt.tight_layout()
    output_comp = OUTPUTS_DIR / "connectivity_comparison_grand_average.png"
    plt.savefig(output_comp, dpi=300)
    print(f"  ✅ Đã lưu: {output_comp}")

    # --- BIỂU ĐỒ 2: DIFFERENCE MATRIX ---
    plt.figure(figsize=(10, 9))
    im_diff = plt.imshow(adj_diff, cmap='RdBu_r', vmin=-0.15, vmax=0.15)
    plt.title("Difference Connectivity Matrix (Incorrect - Correct)", fontsize=14, fontweight='bold')
    
    plt.xticks(np.arange(len(ch_names)), ch_names, rotation=90, fontsize=9)
    plt.yticks(np.arange(len(ch_names)), ch_names, fontsize=9)
    
    plt.colorbar(im_diff, fraction=0.046, pad=0.04, label='$\Delta$ PLV')
    plt.grid(False)
    
    plt.tight_layout()
    output_diff = OUTPUTS_DIR / "connectivity_difference_matrix.png"
    plt.savefig(output_diff, dpi=300)
    print(f"  ✅ Đã lưu: {output_diff}")

if __name__ == "__main__":
    redraw_connectivity_plots()
