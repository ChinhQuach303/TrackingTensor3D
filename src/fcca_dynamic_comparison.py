import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from config import *

def get_fiedler_vector(adj_matrix):
    n = adj_matrix.shape[0]
    # Degree matrix
    d = np.diag(np.sum(adj_matrix, axis=1))
    # Laplacian
    laplacian = d - adj_matrix
    
    try:
        eigenvalues, eigenvectors = eigh(laplacian)
        return eigenvectors[:, 1]
    except Exception as e:
        print(f"  ❌ Lỗi giải trị riêng: {e}")
        return np.zeros(n)

def fiedler_consensus_clustering(connectivity_tensor, time_window):
    n_subs, n_nodes, _, n_times = connectivity_tensor.shape
    start_t, end_t = time_window
    
    W = np.zeros((n_nodes, n_nodes))
    count = 0
    
    time_ms = np.linspace(-1000, 1000, n_times)
    time_indices = np.where((time_ms >= start_t) & (time_ms <= end_t))[0]
    
    for s in range(n_subs):
        # Lấy dữ liệu của subject s: (N, N, T)
        sub_data = connectivity_tensor[s, :, :, :]
        # Lấy cửa sổ thời gian: (N, N, len(time_indices))
        sub_win = sub_data[:, :, time_indices]
        # Trung bình theo trục thời gian (axis=2) -> (N, N)
        adj_sub_win = np.mean(sub_win, axis=2)
        
        f_vec = get_fiedler_vector(adj_sub_win)
        clusters = (f_vec > 0).astype(int)
        
        T_r = (clusters[:, None] == clusters[None, :]).astype(float)
        W += T_r
        count += 1
            
    W /= count
    consensus_f_vec = get_fiedler_vector(W)
    final_clusters = (consensus_f_vec > 0).astype(int)
    return final_clusters, W

def main():
    print("🚀 Bắt đầu Phân tích Dynamic FCCA (Baseline vs ERN)...")
    
    lowrank_path = TENSOR_DIR / "horls_lowrank_incorrect.npy"
    tensor_inc = np.load(lowrank_path)
    
    baseline_win = (-400, -200)
    ern_win = (50, 150)
    
    cls_base, W_base = fiedler_consensus_clustering(tensor_inc, baseline_win)
    cls_ern, W_ern = fiedler_consensus_clustering(tensor_inc, ern_win)
    
    sample_files = list(EPOCHS_DIR.glob("*.fif"))
    epochs = mne.read_epochs(sample_files[0], preload=False, verbose=False)
    ch_names = epochs.ch_names
    info = mne.create_info(ch_names, SFREQ, ch_types='eeg')
    info.set_montage(MONTAGE_NAME)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Consensus Matrices
    im1 = axes[0, 0].imshow(W_base, cmap='YlGnBu', vmin=0.5, vmax=1)
    axes[0, 0].set_title(f"Baseline Consensus Matrix ({baseline_win}ms)")
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(W_ern, cmap='YlGnBu', vmin=0.5, vmax=1)
    axes[0, 1].set_title(f"ERN Window Consensus Matrix ({ern_win}ms)")
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Topomaps
    from mne.viz import plot_topomap
    plot_topomap(cls_base, info, axes=axes[1, 0], show=False, cmap='RdBu_r', sphere=0.1)
    axes[1, 0].set_title("Baseline Clusters (Pre-Response)")
    
    plot_topomap(cls_ern, info, axes=axes[1, 1], show=False, cmap='RdBu_r', sphere=0.1)
    axes[1, 1].set_title("ERN Clusters (Reconfigured)")
    
    plt.tight_layout()
    output_img = OUTPUTS_DIR / "fcca_dynamic_comparison.png"
    plt.savefig(output_img)
    
    mod_base = np.var(W_base)
    mod_ern = np.var(W_ern)
    
    print(f"\n📊 ĐÁNH GIÁ KẾT QUẢ:")
    print(f"  -> Modularity (Baseline): {mod_base:.4f}")
    print(f"  -> Modularity (ERN): {mod_ern:.4f}")
    
    if mod_ern < mod_base:
        print("  🔥 KẾT LUẬN: Mạng lưới trở nên HÒA NHẬP (Integrated) hơn khi có lỗi (ERN).")
    else:
        print("  ⚠️ Mạng lưới giữ nguyên hoặc tăng tính phân tách.")
        
    print(f"\n✅ Kết quả lưu tại: {output_img}")

if __name__ == "__main__":
    main()
