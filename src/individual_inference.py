import numpy as np
import mne
import torch
from pathlib import Path
from config import *
from master_preprocessing import preprocess_subject_ozdemir_style as preprocess_single_subject
from rid_rihaczek import RIDRihaczek
from tensor_decomposition import HORLSDecomposer, find_change_points
import matplotlib.pyplot as plt

def run_full_inference(sub_id):
    print(f"--- BẮT ĐẦU CHẨN ĐOÁN FULL-PIPELINE: {sub_id} ---")
    
    # 1. Load Preprocessed Epochs
    epoch_file = EPOCHS_DIR / f"{sub_id}_master-epo.fif"
    if not epoch_file.exists():
        print(f"  [!] Đang tiền xử lý {sub_id}...")
        epochs = preprocess_single_subject(sub_id)
    else:
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)

    # 2. Compute Connectivity (PLV) using RID-Rihaczek
    print(f"  [2/4] Đang tính toán PLV dải Theta (4-8Hz)...")
    data = epochs['Incorrect'].get_data() # (trials, channels, times)
    n_trials, n_chs, n_times = data.shape
    
    # Extract Theta Band Phase using RID-Rihaczek (optimized placeholder)
    tfd = RIDRihaczek(n_times)
    phases = np.zeros((n_trials, n_chs, n_times))
    for i in range(n_trials):
        phases[i] = tfd.compute_phase(data[i])
        
    # Compute PLV across trials for this subject
    # PLV = |sum(exp(i * delta_phase))| / n_trials
    plv_sub = np.zeros((n_chs, n_chs, n_times))
    for t in range(n_times):
        for i in range(n_chs):
            for j in range(i+1, n_chs):
                phase_diff = phases[:, i, t] - phases[:, j, t]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                plv_sub[i, j, t] = plv
                plv_sub[j, i, t] = plv
    
    # 3. HO-RLS Individual Prediction
    print(f"  [3/4] Đang thực hiện phân rã Tensor & Tìm Change-points...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sub_tensor_4d = torch.tensor(plv_sub, dtype=torch.float32, device=device).unsqueeze(0) # (1, 30, 30, 256)
    
    decomposer = HORLSDecomposer(n_nodes=n_chs, device=device)
    w_indiv, energy_indiv = decomposer.decompose(sub_tensor_4d)
    
    # 4. Results & Visualization
    n_samples = energy_indiv.shape[0]
    cp_indices = find_change_points(energy_indiv)
    time_ms = np.linspace(-1000, 1000, n_samples)
    cp_times = time_ms[cp_indices]
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_ms, energy_indiv, color='blue', label=f'Individual Dynamics ({sub_id})')
    
    # Group reference (Ozdemir 25ms)
    plt.axvline(25, color='black', linestyle=':', label='Group Master CP (25ms)')
    
    found_any = False
    for cp in cp_times:
        if 0 < cp < 500:
            plt.axvline(cp, color='red', linestyle='--', label=f'Personal CP: {cp:.0f}ms')
            print(f"  [KẾT QUẢ]: Phát hiện Change-point tại {cp:.1f} ms")
            found_any = True
            
    if not found_any:
        print("  [!] Không phát hiện Change-point rõ rệt trong cửa sổ ERN cho đối tượng này.")

    plt.title(f"Diagnostic Report for Excluded Subject: {sub_id} ({n_trials} trials)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Network Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    diag_img = OUTPUTS_DIR / f"test_excluded_{sub_id}.png"
    plt.savefig(diag_img)
    print(f"--- Hoàn tất. Kết quả lưu tại: {diag_img} ---")

if __name__ == "__main__":
    # Test sub-007 (14 trials) and sub-005 (2 trials)
    for sid in ["sub-007", "sub-005"]:
        run_full_inference(sid)
