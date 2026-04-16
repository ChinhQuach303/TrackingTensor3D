import numpy as np
import mne
import torch
from pathlib import Path
from config import *
from master_preprocessing import preprocess_subject_ozdemir_style as preprocess_single_subject
from rid_rihaczek import RIDRihaczek
from tensor_decomposition import HORLSDecomposer, find_change_points
import matplotlib.pyplot as plt

def run_individual_diagnostic(sub_id):
    print(f"--- BẮT ĐẦU CHẨN ĐOÁN CHO SUBJECT: {sub_id} ---")
    
    # 1. Preprocess if epochs don't exist
    epoch_file = EPOCHS_DIR / f"{sub_id}_master-epo.fif"
    if not epoch_file.exists():
        print("1. Đang tiền xử lý (Filtering, ICA, CSD)...")
        epochs = preprocess_single_subject(sub_id)
    else:
        epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)

    # 2. Get Tensor (In this demo, we assume the tensor is pre-calculated or we simulate)
    # In a real pipeline, you'd call the connectivity engine here.
    # We'll use the 4D tensor if it exists for this subject.
    if TENSOR_INCORRECT_FILE.exists():
        full_tensor = np.load(TENSOR_INCORRECT_FILE)
        # Find which index corresponds to this subject
        # For simplicity in demo, we'll just use the first 4D slice as "the subject"
        sub_tensor_4d = full_tensor[0:1, :, :, :] # (1, 30, 30, 256)
    else:
        print("Lỗi: Không tìm thấy dữ liệu Tensor 4D để thực hiện suy luận.")
        return

    # 3. HO-RLS Individual Prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_nodes = sub_tensor_4d.shape[1]
    
    decomposer = HORLSDecomposer(n_nodes=n_nodes, device=device)
    w_indiv, energy_indiv = decomposer.decompose(torch.tensor(sub_tensor_4d, dtype=torch.float32, device=device))
    
    # 4. Detect Change-points
    cp_indices = find_change_points(energy_indiv)
    time_ms = np.linspace(-1000, 1000, 256)
    cp_times = time_ms[cp_indices]
    
    # 5. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(time_ms, energy_indiv, color='green', label=f'Individual Dynamics for {sub_id}')
    
    for cp in cp_times:
        if 0 < cp < 300: # Focus on the ERN window
            plt.axvline(cp, color='red', linestyle='--', label=f'Detected CP: {cp:.0f}ms')
            print(f"  [KẾT QUẢ]: Phát hiện Change-point tại {cp:.1f} ms")

    plt.title(f"Individual Diagnostic Report: {sub_id}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Network Energy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    diag_img = OUTPUTS_DIR / f"diagnostic_{sub_id}_final.png"
    plt.savefig(diag_img)
    print(f"--- Chẩn đoán hoàn tất. Báo cáo lưu tại: {diag_img} ---")

if __name__ == "__main__":
    # Test on sub-001
    run_individual_diagnostic("sub-001")
