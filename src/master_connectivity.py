import mne
import numpy as np
import torch
from pathlib import Path
import pandas as pd

def rid_rihaczek_gpu(data_tensor, sigma=1.0):
    n_trials, N = data_tensor.shape
    device = data_tensor.device
    
    # 1. Local Autocorrelation
    R = torch.zeros((n_trials, N, N), dtype=torch.complex64, device=device)
    for tau in range(-(N // 2), N // 2 + (N % 2)):
        col_idx = tau + N // 2
        if col_idx < 0 or col_idx >= N: continue
        
        if tau >= 0:
            real_part = data_tensor[:, tau:] * data_tensor[:, :N-tau]
            R[:, tau:, col_idx] = torch.complex(real_part, torch.zeros_like(real_part))
        else:
            real_part = data_tensor[:, :N+tau] * data_tensor[:, -tau:]
            R[:, :N+tau, col_idx] = torch.complex(real_part, torch.zeros_like(real_part))

    # 2. Kernel Gaussian
    t_grid = torch.arange(N, device=device).float() - N // 2
    tau_grid = torch.arange(N, device=device).float() - N // 2
    tt, tau_tau = torch.meshgrid(t_grid, tau_grid, indexing='ij')
    tau_tau_safe = torch.where(tau_tau == 0, 1e-9, tau_tau)
    kernel = torch.sqrt(sigma / (4 * np.pi * tau_tau_safe**2)) * torch.exp(-sigma * (tt**2) / (4 * tau_tau_safe**2))
    kernel /= (torch.sum(kernel, axis=0) + 1e-9)

    # 3. Convolution
    R_flat_real = R.real.permute(0, 2, 1).reshape(n_trials * N, 1, N)
    R_flat_imag = R.imag.permute(0, 2, 1).reshape(n_trials * N, 1, N)
    kernel_flip = kernel.T.flip(1).unsqueeze(1)
    
    R_filt_real = torch.zeros_like(R_flat_real)
    R_filt_imag = torch.zeros_like(R_flat_imag)
    
    for j in range(N):
        k = kernel_flip[j:j+1]
        R_filt_real[j::N] = torch.nn.functional.conv1d(R_flat_real[j::N], k, padding=N//2)[:, :, :N]
        R_filt_imag[j::N] = torch.nn.functional.conv1d(R_flat_imag[j::N], k, padding=N//2)[:, :, :N]

    R_final = torch.complex(R_filt_real, R_filt_imag).reshape(n_trials, N, N).permute(0, 2, 1)
    
    # 4. FFT
    C = torch.fft.fft(R_final, dim=2)
    return C

def compute_master_tensors():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    refined_dir = base_path / "data" / "processed" / "refined_master"
    tensor_dir = base_path / "data" / "processed" / "connectivity"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    
    sub_files = sorted(list(refined_dir.glob("*_theta_balanced-epo.fif")))
    n_subs = len(sub_files)
    
    # Initial load to get metadata
    sample_epochs = mne.read_epochs(sub_files[0], preload=False)
    n_channels = len(sample_epochs.ch_names)
    n_times = len(sample_epochs.times)
    sfreq = sample_epochs.info['sfreq']
    
    # Pre-calculate theta mask
    freq_axis = np.fft.fftfreq(n_times, 1/sfreq)
    theta_mask = (freq_axis >= 4) & (freq_axis <= 8)
    
    # Output Tensors: (Subject, Channel, Channel, 256)
    tensor_correct = np.zeros((n_subs, n_channels, n_channels, 256), dtype=np.float32)
    tensor_incorrect = np.zeros((n_subs, n_channels, n_channels, 256), dtype=np.float32)
    
    print(f"Bắt đầu Giai đoạn 3: Tính Connectivity cho {n_subs} subjects (Time=256) trên {device}...")
    
    for i, f in enumerate(sub_files):
        sub_name = f.stem.split('_')[0]
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        for cond, output_tensor in zip(['Correct', 'Incorrect'], [tensor_correct, tensor_incorrect]):
            # EEG data in microvolts for numerical stability
            # Epochs have 257 points, we take first 256
            raw_data = epochs[cond].get_data()[:, :, :256] * 1e6
            data = torch.tensor(raw_data, dtype=torch.float32, device=device)
            n_trials = data.shape[0]
            
            # Phase extraction
            theta_complex = torch.zeros((n_trials, n_channels, 256), dtype=torch.complex64, device=device)
            # Adjust theta mask for 256 points
            freq_axis_256 = np.fft.fftfreq(256, 1/sfreq)
            theta_mask_256 = (freq_axis_256 >= 4) & (freq_axis_256 <= 8)
            
            for ch in range(n_channels):
                C = rid_rihaczek_gpu(data[:, ch, :])
                theta_complex[:, ch, :] = torch.mean(C[:, :, theta_mask_256], dim=2)
            
            # Normalize to unit vectors (Phase only)
            theta_complex /= (torch.abs(theta_complex) + 1e-12)
            
            # PLV calculation
            for t in range(256):
                Z_t = theta_complex[:, :, t] # (n_trials, n_channels)
                plv = torch.abs(torch.matmul(Z_t.H, Z_t)) / n_trials
                output_tensor[i, :, :, t] = plv.cpu().numpy()
                
        print(f"  DONE: {sub_name} ({i+1}/{n_subs})")
        
    # Final Savings
    np.save(tensor_dir / "tensor_correct_4d.npy", tensor_correct)
    np.save(tensor_dir / "tensor_incorrect_4d.npy", tensor_incorrect)
    print("-" * 50)
    print(f"HOÀN TẤT GIAI ĐOẠN 3! Tensors lưu tại: {tensor_dir}")

if __name__ == "__main__":
    compute_master_tensors()
