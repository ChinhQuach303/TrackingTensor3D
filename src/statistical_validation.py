import numpy as np
import torch
import matplotlib.pyplot as plt
from tensor_decomposition import HORLSDecomposer
from config import *

def run_statistical_validation(n_permutations=100):
    print(f"🧪 Bắt đầu Kiểm định hoán vị (Permutation Testing) với {n_permutations} lần lặp...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load dữ liệu đã cân bằng (Balanced Tensors)
    tensor_corr_raw = np.load(TENSOR_CORRECT_FILE) # (S, N, N, T)
    tensor_inc_raw = np.load(TENSOR_INCORRECT_FILE)
    n_subs, n_nodes, _, n_times = tensor_corr_raw.shape
    
    # 2. Tính năng lượng thực tế (Observed Difference)
    decomposer = HORLSDecomposer(n_nodes=n_nodes, n_subs=n_subs, device=device)
    
    print("  -> Tính năng lượng thực tế...")
    _, energy_corr_obs, _ = decomposer.decompose(torch.tensor(tensor_corr_raw, device=device))
    _, energy_inc_obs, _ = decomposer.decompose(torch.tensor(tensor_inc_raw, device=device))
    observed_diff = energy_inc_obs - energy_corr_obs
    
    # 3. Chạy Permutation Loop
    null_diffs = []
    print(f"  -> Đang chạy {n_permutations} lần hoán vị trên {device}...")
    
    for i in range(n_permutations):
        # Ngẫu nhiên tráo đổi nhãn Correct/Incorrect cho từng subject
        swap_mask = np.random.choice([True, False], size=n_subs)
        
        perm_corr = tensor_corr_raw.copy()
        perm_inc = tensor_inc_raw.copy()
        
        # Tráo đổi các lát cắt (slices) của subject
        perm_corr[swap_mask] = tensor_inc_raw[swap_mask]
        perm_inc[swap_mask] = tensor_corr_raw[swap_mask]
        
        # Tính năng lượng cho bộ dữ liệu đã tráo
        _, e_corr_perm, _ = decomposer.decompose(torch.tensor(perm_corr, device=device))
        _, e_inc_perm, _ = decomposer.decompose(torch.tensor(perm_inc, device=device))
        
        null_diffs.append(e_inc_perm - e_corr_perm)
        
        if (i + 1) % 20 == 0:
            print(f"     Tiến độ: {i+1}/{n_permutations}...")

    null_diffs = np.array(null_diffs)
    
    # 4. Tính p-value cho từng mốc thời gian
    # p-value = (Số lần null_diff >= observed_diff) / n_permutations
    p_values = np.sum(np.abs(null_diffs) >= np.abs(observed_diff), axis=0) / n_permutations
    
    # 5. Vẽ biểu đồ kết quả thống kê
    time_ms = np.linspace(-1000, 1000, n_times)
    plt.figure(figsize=(14, 7))
    
    # Vẽ vùng Significant (p < 0.05)
    sig_mask = p_values < 0.05
    plt.fill_between(time_ms, observed_diff, where=sig_mask, color='red', alpha=0.3, label='Significant Difference (p < 0.05)')
    
    plt.plot(time_ms, observed_diff, color='black', linewidth=2, label='Observed Difference (Inc - Cor)')
    plt.axvline(x=0, color='blue', linestyle='--', label='Response')
    
    plt.title("Statistical Validation: Network Energy Difference (Permutation Test)")
    plt.xlabel("Time (ms)")
    plt.ylabel("$\Delta$ Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = OUTPUTS_DIR / "statistical_validation_final.png"
    plt.savefig(output_path)
    print(f"\n✅ HOÀN TẤT KIỂM ĐỊNH!")
    print(f"📊 Biểu đồ thống kê lưu tại: {output_path}")
    
    # Lưu kết quả p-values
    np.save(TENSOR_DIR / "permutation_p_values.npy", p_values)

if __name__ == "__main__":
    run_statistical_validation(n_permutations=100)
