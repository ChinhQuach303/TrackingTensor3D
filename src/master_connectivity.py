import mne
import numpy as np
import torch
import torch.fft
from pathlib import Path
from config import *

def rid_rihaczek_gpu(signal_tensor):
    n_trials, n_points = signal_tensor.shape
    device = signal_tensor.device
    
    tau = torch.arange(-n_points // 2, n_points // 2, device=device)
    theta = torch.fft.fftfreq(n_points, d=1.0, device=device) * 2 * np.pi
    
    THETA, TAU = torch.meshgrid(theta, tau, indexing='ij')
    
    sigma = 1.0 
    kernel = torch.exp(-(THETA * TAU)**2 / sigma) * torch.exp(1j * THETA * TAU / 2)
    
    # Ambiguity Function via Cyclic Autocorrelation
    AF = torch.zeros((n_trials, n_points, n_points), dtype=torch.complex64, device=device)
    for i, t in enumerate(tau):
        sh = t.item()
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

def run_connectivity_pipeline():
    sub_files = list(REFINED_DIR.glob("*_theta_balanced-epo.fif"))
    sub_files.sort()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get metadata from first file
    sample_epochs = mne.read_epochs(sub_files[0], preload=False, verbose=False)
    n_channels = len(sample_epochs.ch_names)
    n_subs = len(sub_files)
    
    # Output Tensors: (Subject, Channel, Channel, 256)
    tensor_correct = np.zeros((n_subs, n_channels, n_channels, 256), dtype=np.float32)
    tensor_incorrect = np.zeros((n_subs, n_channels, n_channels, 256), dtype=np.float32)
    
    print(f"Bắt đầu Giai đoạn 3: Tính Connectivity cho {n_subs} subjects (Time=256) trên {device}...")
    
    for i, f in enumerate(sub_files):
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        for cond, output_tensor in zip(['Correct', 'Incorrect'], [tensor_correct, tensor_incorrect]):
            raw_data = epochs[cond].get_data()[:, :, :256] * 1e6
            data = torch.tensor(raw_data, dtype=torch.float32, device=device)
            n_trials = data.shape[0]
            
            # Phase extraction
            theta_complex = torch.zeros((n_trials, n_channels, 256), dtype=torch.complex64, device=device)
            freq_axis_256 = np.fft.fftfreq(256, 1/SFREQ)
            theta_mask_256 = (freq_axis_256 >= THETA_BAND[0]) & (freq_axis_256 <= THETA_BAND[1])
            
            for ch in range(n_channels):
                C = rid_rihaczek_gpu(data[:, ch, :])
                theta_complex[:, ch, :] = torch.mean(C[:, :, theta_mask_256], dim=2)
            
            theta_complex /= (torch.abs(theta_complex) + 1e-12)
            
            for t in range(256):
                Z_t = theta_complex[:, :, t]
                plv = torch.abs(torch.matmul(Z_t.H, Z_t)) / n_trials
                output_tensor[i, :, :, t] = plv.cpu().numpy()
        
        # print(f"  DONE: {f.stem.split('_')[0]} ({i+1}/{n_subs})")

    np.save(TENSOR_CORRECT_FILE, tensor_correct)
    np.save(TENSOR_INCORRECT_FILE, tensor_incorrect)
    print("-" * 50)
    print(f"HOÀN TẤT GIAI ĐOẠN 3! Tensors lưu tại: {TENSOR_DIR}")

if __name__ == "__main__":
    run_connectivity_pipeline()
