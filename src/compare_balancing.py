import mne
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import *
from master_connectivity import rid_rihaczek_gpu, temporal_matching_subsample

def compare_balancing_effect(sub_id='sub-006'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_file = EPOCHS_DIR / f"{sub_id}_master-epo.fif"
    epochs = mne.read_epochs(epoch_file, preload=True, verbose=False)
    
    # 1. Lấy dữ liệu Unbalanced (Toàn bộ Correct)
    epochs_cor_all = epochs['Correct']
    n_cor_all = len(epochs_cor_all)
    
    # 2. Lấy dữ liệu Balanced (Temporal Matched)
    _, epochs_cor_bal = temporal_matching_subsample(epochs)
    n_cor_bal = len(epochs_cor_bal)
    
    # 3. Lấy dữ liệu Incorrect
    epochs_inc = epochs['Incorrect']
    n_inc = len(epochs_inc)
    
    results = {}
    for label, ep_obj in [('Correct_Unbalanced', epochs_cor_all), 
                          ('Correct_Balanced', epochs_cor_bal), 
                          ('Incorrect', epochs_inc)]:
        
        print(f"🔄 Đang tính PLV cho {label} ({len(ep_obj)} trials)...")
        data = torch.tensor(ep_obj.get_data()[:, :, :256] * 1e6, dtype=torch.float32, device=device)
        n_trials, n_channels, _ = data.shape
        
        theta_complex = torch.zeros((n_trials, n_channels, 256), dtype=torch.complex64, device=device)
        freq_axis = np.fft.fftfreq(256, 1/SFREQ)
        theta_mask = (freq_axis >= THETA_BAND[0]) & (freq_axis <= THETA_BAND[1])
        
        for ch in range(n_channels):
            C = rid_rihaczek_gpu(data[:, ch, :])
            theta_complex[:, ch, :] = torch.mean(C[:, :, theta_mask], dim=2)
        
        theta_complex /= (torch.abs(theta_complex) + 1e-12)
        
        # Tính Global Mean PLV
        global_plv = torch.zeros(256, device=device)
        for t in range(256):
            Z_t = theta_complex[:, :, t]
            plv_matrix = torch.abs(torch.matmul(Z_t.H, Z_t)) / n_trials
            global_plv[t] = torch.mean(plv_matrix)
            
        results[label] = global_plv.cpu().numpy()

    # --- VẼ BIỂU ĐỒ SO SÁNH ---
    plt.figure(figsize=(12, 7))
    time_ms = np.linspace(-1000, 1000, 256)
    
    plt.plot(time_ms, results['Correct_Unbalanced'], label=f'Correct UNBALANCED ({n_cor_all} trials)', color='blue', linestyle='--')
    plt.plot(time_ms, results['Correct_Balanced'], label=f'Correct BALANCED ({n_cor_bal} trials)', color='blue', linewidth=2)
    plt.plot(time_ms, results['Incorrect'], label=f'Incorrect ({n_inc} trials)', color='red', linewidth=2)
    
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title(f"So sánh hiệu ứng Cân bằng Trials (Subject: {sub_id})")
    plt.xlabel("Time (ms)")
    plt.ylabel("Global Mean PLV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = OUTPUTS_DIR / "balancing_effect_comparison.png"
    plt.savefig(output_img)
    print(f"✅ Biểu đồ so sánh đã lưu tại: {output_img}")

if __name__ == "__main__":
    compare_balancing_effect()
