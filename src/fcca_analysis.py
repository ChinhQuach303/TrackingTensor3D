import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pathlib import Path
from config import *

def get_fiedler_vector(adj_matrix):
    """
    Tính Fiedler vector (eigenvector tương ứng với eigenvalue nhỏ thứ 2)
    của ma trận Laplacian từ ma trận kề adj_matrix.
    """
    n = adj_matrix.shape[0]
    # Degree matrix
    d = np.diag(np.sum(adj_matrix, axis=1))
    # Laplacian
    laplacian = d - adj_matrix
    
    # Giải bài toán trị riêng (eigh cho ma trận đối xứng)
    eigenvalues, eigenvectors = eigh(laplacian)
    
    # Lấy eigenvector tương ứng với eigenvalue nhỏ thứ 2 (index 1)
    # Lưu ý: eigenvalue nhỏ nhất thường là 0 (hoặc xấp xỉ 0)
    fiedler_vec = eigenvectors[:, 1]
    return fiedler_vec

def fiedler_consensus_clustering(connectivity_tensor, time_window):
    """
    Cài đặt Fiedler Consensus Clustering Algorithm (FCCA) theo Ozdemir (2017).
    """
    n_subs, n_nodes, _, n_times = connectivity_tensor.shape
    start_t, end_t = time_window
    
    # Ma trận Co-occurrence W
    W = np.zeros((n_nodes, n_nodes))
    count = 0
    
    print(f"  Đang chạy FCCA trên cửa sổ thời gian {start_t}ms đến {end_t}ms...")
    
    # Duyệt qua từng subject và từng thời điểm trong cửa sổ
    # (Trong thực tế có thể lấy trung bình theo thời gian trước để giảm tải nếu tensor quá lớn)
    time_indices = np.where((np.linspace(-1000, 1000, n_times) >= start_t) & 
                            (np.linspace(-1000, 1000, n_times) <= end_t))[0]
    
    for s in range(n_subs):
        for t in time_indices:
            adj = connectivity_tensor[s, :, :, t]
            
            # 1. Tìm Fiedler vector cho từng mạng đơn lẻ
            f_vec = get_fiedler_vector(adj)
            
            # 2. Bi-partition dựa trên dấu của Fiedler vector
            clusters = (f_vec > 0).astype(int)
            
            # 3. Cập nhật Co-occurrence matrix T_r
            # Tr(i,j) = 1 nếu i và j cùng cluster, 0 nếu khác
            T_r = (clusters[:, None] == clusters[None, :]).astype(float)
            W += T_r
            count += 1
            
    # 4. Tính xác suất đồng xuất hiện (Consensus Matrix)
    W /= count
    
    # 5. Tìm Fiedler vector của ma trận Consensus W
    consensus_f_vec = get_fiedler_vector(W)
    final_clusters = (consensus_f_vec > 0).astype(int)
    
    return final_clusters, W

def main():
    print("Khởi động Phân tích FCCA (Fiedler Consensus Clustering)...")
    
    # Load Low-rank Connectivity Tensor (Đã được khử nhiễu bởi HO-RLSL)
    lowrank_file = TENSOR_DIR / "horls_lowrank_incorrect.npy"
    if not lowrank_file.exists():
        print("Lỗi: Không tìm thấy file tensor low-rank. Vui lòng chạy tensor_decomposition.py trước.")
        return
        
    tensor_inc = np.load(lowrank_file)
    
    # Cửa sổ thời gian ERN (thường từ 0ms đến 150ms sau phản ứng)
    ern_window = (0, 150) 
    
    clusters, W_consensus = fiedler_consensus_clustering(tensor_inc, ern_window)
    
    # Lấy tên kênh từ epoch mẫu
    sample_epo = mne.read_epochs(list(EPOCHS_DIR.glob("*.fif"))[0], preload=False, verbose=False)
    ch_names = sample_epo.ch_names
    
    # Hiển thị kết quả phân cụm
    print("\nKẾT QUẢ PHÂN CỤM (FCCA):")
    cluster_0 = [ch_names[i] for i, c in enumerate(clusters) if c == 0]
    cluster_1 = [ch_names[i] for i, c in enumerate(clusters) if c == 1]
    
    print(f"Cluster A: {', '.join(cluster_0)}")
    print(f"Cluster B: {', '.join(cluster_1)}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Co-occurrence Matrix
    im = axes[0].imshow(W_consensus, cmap='YlGnBu')
    axes[0].set_title("Consensus Co-occurrence Matrix (W)")
    plt.colorbar(im, ax=axes[0])
    
    # 2. Topomap of clusters
    from mne.viz import plot_topomap
    info = mne.create_info(ch_names, SFREQ, ch_types='eeg')
    info.set_montage(MONTAGE_NAME)
    
    # Màu sắc cho 2 cụm
    plot_topomap(clusters, info, axes=axes[1], show=False, cmap='RdBu_r', sphere=0.1)
    axes[1].set_title("FCCA Consensus Clusters (Red vs Blue)")
    
    plt.tight_layout()
    output_img = OUTPUTS_DIR / "fcca_ozdemir_results.png"
    plt.savefig(output_img)
    print(f"\nKết quả FCCA đã được lưu tại: {output_img}")

if __name__ == "__main__":
    main()
