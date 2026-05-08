import mne
import numpy as np
import torch
import torch.fft
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat
from config import *

def rid_rihaczek_gpu(signal_tensor):
    """Tính phân phối RID-Rihaczek trên GPU."""
    n_trials, n_points = signal_tensor.shape
    device = signal_tensor.device
    
    tau = torch.arange(-n_points // 2, n_points // 2, device=device, dtype=torch.float32)
    theta = torch.fft.fftfreq(n_points, d=1.0, device=device) * 2 * np.pi
    
    THETA, TAU = torch.meshgrid(theta, tau, indexing='ij')
    kernel = torch.exp(-(THETA * TAU)**2 / 1.0) * torch.exp(1j * THETA * TAU / 2)
    
    AF = torch.zeros((n_trials, n_points, n_points), dtype=torch.complex64, device=device)
    for i, t in enumerate(tau):
        sh = int(t.item())
        if sh >= 0:
            R = signal_tensor[:, sh:] * torch.conj(signal_tensor[:, :n_points-sh])
            AF[:, :, i] = torch.nn.functional.pad(R, (0, int(abs(sh))))
        else:
            R = signal_tensor[:, :n_points+sh] * torch.conj(signal_tensor[:, int(abs(sh)):])
            AF[:, :, i] = torch.nn.functional.pad(R, (int(abs(sh)), 0))

    AF_theta = torch.fft.fft(AF, dim=1)
    RID_theta = AF_theta * kernel.unsqueeze(0)
    RID_t_f = torch.fft.ifft(torch.fft.ifft(RID_theta, dim=1), dim=2)
    return RID_t_f

def temporal_matching_subsample(epochs):
    """
    Cân bằng số lượng trials Correct và Incorrect bằng cách khớp theo thời gian (Temporal Matching).
    Đảm bảo cả hai nhóm có cùng trạng thái sinh học (mệt mỏi, drift tín hiệu).
    """
    n_inc = len(epochs['Incorrect'])
    n_cor = len(epochs['Correct'])
    
    if n_inc < MIN_INCORRECT_TRIALS:
        return None
        
    # Lấy thời điểm xuất hiện của các trials
    times_inc = epochs['Incorrect'].events[:, 0]
    times_cor = epochs['Correct'].events[:, 0]
    
    selected_cor_indices = []
    available_cor_indices = list(range(n_cor))
    
    for t_i in times_inc:
        # Tìm trial Correct có thời gian gần t_i nhất
        diffs = np.abs(times_cor[available_cor_indices] - t_i)
        best_match_idx_in_available = np.argmin(diffs)
        
        # Lưu index thực tế
        real_idx = available_cor_indices.pop(best_match_idx_in_available)
        selected_cor_indices.append(real_idx)
        
    # Tạo Epochs mới đã được cân bằng
    epochs_inc = epochs['Incorrect']
    epochs_cor_matched = epochs['Correct'][selected_cor_indices]
    
    return epochs_inc, epochs_cor_matched

def run_connectivity_pipeline():
    sub_files = sorted(list(EPOCHS_DIR.glob("*_master-epo.fif")))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    valid_subs = []
    # Test metadata
    sample_epochs = mne.read_epochs(sub_files[0], preload=False, verbose=False)
    n_channels = len(sample_epochs.ch_names)
    
    # List lưu trữ tạm thời
    all_corr = []
    all_inc = []
    processed_subs = []

    print(f"🚀 Chạy Connectivity với TEMPORAL MATCHING trên {device}...")
    
    for i, f in enumerate(sub_files):
        sid = f.name.split('_')[0]
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        # Thực hiện cân bằng dữ liệu
        balanced = temporal_matching_subsample(epochs)
        if balanced is None:
            print(f"  ⚠️ {sid} bị loại do số lần sai < {MIN_INCORRECT_TRIALS}")
            continue
            
        processed_subs.append(sid)
        epochs_inc, epochs_cor = balanced
        
        # Kết quả PLV cho 1 subject
        sub_plv_cor = np.zeros((n_channels, n_channels, 256), dtype=np.float32)
        sub_plv_inc = np.zeros((n_channels, n_channels, 256), dtype=np.float32)

        for cond_label, ep_obj, output_arr in [('Correct', epochs_cor, sub_plv_cor), ('Incorrect', epochs_inc, sub_plv_inc)]:
            raw_data = ep_obj.get_data()[:, :, :256] * 1e6
            data = torch.tensor(raw_data, dtype=torch.float32, device=device)
            n_trials = data.shape[0]
            
            theta_complex = torch.zeros((n_trials, n_channels, 256), dtype=torch.complex64, device=device)
            freq_axis = np.fft.fftfreq(256, 1/SFREQ)
            theta_mask = (freq_axis >= THETA_BAND[0]) & (freq_axis <= THETA_BAND[1])
            
            for ch in range(n_channels):
                C = rid_rihaczek_gpu(data[:, ch, :])
                theta_complex[:, ch, :] = torch.mean(C[:, :, theta_mask], dim=2)
            
            theta_complex /= (torch.abs(theta_complex) + 1e-12)
            
            for t in range(256):
                Z_t = theta_complex[:, :, t]
                plv = torch.abs(torch.matmul(Z_t.H, Z_t)) / n_trials
                output_arr[:, :, t] = plv.cpu().numpy()
        
        all_corr.append(sub_plv_cor)
        all_inc.append(sub_plv_inc)
        print(f"  Done: {sid} (Balanced with {len(epochs_inc)} trials)")

    # Chuyển về Tensor 4D (Subject, Node, Node, Time)
    tensor_correct = np.stack(all_corr, axis=0)
    tensor_incorrect = np.stack(all_inc, axis=0)

    # Lưu kết quả
    np.save(TENSOR_CORRECT_FILE, tensor_correct)
    np.save(TENSOR_INCORRECT_FILE, tensor_incorrect)
    
    # MATLAB export
    savemat(TENSOR_DIR / "connectivity_balanced_4d.mat", {
        'tensor_correct': tensor_correct,
        'tensor_incorrect': tensor_incorrect,
        'subjects': processed_subs,
        'channels': sample_epochs.ch_names
    })
    
    print(f"\n✅ HOÀN TẤT! Đã xử lý {len(processed_subs)} subjects thành công.")
    print(f"📦 Dữ liệu đã được cân bằng hoàn toàn (Temporal Matched).")

if __name__ == "__main__":
    run_connectivity_pipeline()
