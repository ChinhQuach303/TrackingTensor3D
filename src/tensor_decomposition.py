import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

class HORLSDecomposer:
    def __init__(self, n_nodes, alpha=8.0, sigma=0.11, device='cpu'):
        self.N = n_nodes
        self.alpha = alpha
        self.sigma = sigma
        self.device = device
        
    def tucker_init(self, tensor_4d, n_init=10):
        """
        Khởi tạo subspace bằng Tucker Decomposition trên n_init điểm đầu tiên (baseline).
        Theo Ozdemir (2017), tucker decomposition cung cấp basis ban đầu cho recursive tracking.
        """
        # M_train: (Subs, Chan, Chan, n_init) -> Mean across subs: (Chan, Chan, n_init)
        m_train = torch.mean(tensor_4d[:, :, :, :n_init], dim=0)
        
        # Unfold along Mode-1 (Channel mode)
        # M1: (Chan, Chan * n_init)
        m1 = m_train.reshape(self.N, -1)
        
        # SVD to find the dominant subspace
        u, s, v = torch.svd(m1)
        
        # w0 is the first singular vector (dominant direction)
        w0 = u[:, 0] 
        return w0

    def decompose(self, tensor_4d):
        n_subs, _, _, n_times = tensor_4d.shape
        
        # --- REFACTORED: Tucker Initialization (First 10 points) ---
        w = self.tucker_init(tensor_4d, n_init=10)
        P = torch.eye(self.N, device=self.device) * (1.0 / self.sigma)
        
        weights_evolution = torch.zeros((self.N, n_times), device=self.device)
        energy = torch.zeros(n_times, device=self.device)
        
        obs_tensor = torch.mean(tensor_4d, dim=0)
        
        for t in range(n_times):
            C_t = obs_tensor[:, :, t]
            
            # HO-RLS Logic (Recursive Subspace Tracking)
            y_t = torch.matmul(C_t, w)
            Pi_t = torch.matmul(P, y_t)
            k_t = Pi_t / (self.alpha + torch.dot(y_t, Pi_t))
            
            # Update w using recursive relation
            w_new = torch.matmul(C_t, w)
            w_new /= (torch.norm(w_new) + 1e-12)
            
            w = w_new
            P = (P - torch.outer(k_t, Pi_t)) / self.alpha
            weights_evolution[:, t] = w
            energy[t] = torch.norm(torch.matmul(C_t, w)) # Network Energy
            
        return weights_evolution.cpu().numpy(), energy.cpu().numpy()

def find_change_points(energy, threshold_factor=1.5):
    # Detect abrupt changes in energy derivative
    diff_energy = np.abs(np.diff(energy))
    threshold = np.mean(diff_energy) + threshold_factor * np.std(diff_energy)
    cp_indices = np.where(diff_energy > threshold)[0]
    
    # Filter to keep only distinct points (not consecutive)
    if len(cp_indices) == 0: return []
    
    distinct_cp = [cp_indices[0]]
    for idx in cp_indices[1:]:
        if idx - distinct_cp[-1] > 20: # At least 20 samples apart (~150ms)
            distinct_cp.append(idx)
    return distinct_cp

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Refining Phase 4 Analysis on {device} to match Ozdemir Paper...")
    
    # Process both conditions
    for cond, tensor_file in zip(['incorrect', 'correct'], [TENSOR_INCORRECT_FILE, TENSOR_CORRECT_FILE]):
        print(f"  Processing {cond} condition...")
        data = np.load(tensor_file)
        tensor = torch.tensor(data, dtype=torch.float32, device=device)
        n_nodes = tensor.shape[1]
        
        decomposer = HORLSDecomposer(n_nodes=n_nodes, device=device)
        w_t, energy = decomposer.decompose(tensor)
        
        # Save results for statistical validation
        np.save(TENSOR_DIR / f"horls_weights_{cond}.npy", w_t)
        np.save(TENSOR_DIR / f"horls_energy_{cond}.npy", energy)
        
        if cond == 'incorrect':
            # Find Change-points for the Incorrect condition (main focus of the paper)
            cp_indices = find_change_points(energy)
            time_ms = np.linspace(-1000, 1000, 256)
            cp_times = time_ms[cp_indices]
            
            # --- PLOTTING LIKE OZDEMIR ---
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1])
            
            # 1. Network Energy and Change-points
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(time_ms, energy, color='black', linewidth=2, label='Common Network Energy')
            for cp in cp_times:
                ax0.axvline(cp, color='red', linestyle='--', alpha=0.8, label='Change-point' if cp == cp_times[0] else None)
            ax0.set_title("Network Dynamics & Detected Change-points (Incorrect Condition)")
            ax0.set_ylabel("Energy")
            ax0.legend()
            ax0.grid(True, alpha=0.3)
            
            # 2. Heatmap of Weights (Channel x Time)
            ax1 = fig.add_subplot(gs[1])
            sample_epo_files = list(EPOCHS_DIR.glob("*.fif"))
            if not sample_epo_files:
                print("Lỗi: Không tìm thấy tệp epoch để lấy tên kênh.")
                return
            sample_epo = mne.read_epochs(sample_epo_files[0], preload=False, verbose=False)
            ch_names = sample_epo.ch_names
            
            im = ax1.imshow(w_t, aspect='auto', extent=[-1000, 1000, n_nodes, 0], cmap='jet')
            ax1.set_yticks(np.arange(n_nodes) + 0.5)
            ax1.set_yticklabels(ch_names, fontsize=8)
            ax1.set_title("Time-Varying Network Weights ($w_t$)")
            ax1.set_xlabel("Time (ms)")
            ax1.axvline(0, color='white', linestyle='-', linewidth=2)
            plt.colorbar(im, ax=ax1, label='Weight Amplitude')
            
            # 3. Topomap at peak ERN (around 50-70ms)
            ax2 = fig.add_subplot(gs[2])
            # Find max energy point in 25-100ms
            ern_mask = (time_ms >= 25) & (time_ms <= 100)
            peak_idx = np.where(ern_mask)[0][np.argmax(energy[ern_mask])]
            peak_time = time_ms[peak_idx]
            
            info = mne.create_info(ch_names, SFREQ, ch_types='eeg')
            info.set_montage(MONTAGE_NAME)
            
            from mne.viz import plot_topomap
            im_topo, _ = plot_topomap(w_t[:, peak_idx], info, axes=ax2, show=False, cmap='Reds', sphere=0.1)
            ax2.set_title(f"Network Topology at Peak ERN ({peak_time:.1f} ms)")
            
            plt.tight_layout()
            output_img = OUTPUTS_DIR / "ozdemir_replicated_dynamics.png"
            plt.savefig(output_img)
            print(f"Refined Paper-Style plot saved to: {output_img}")

if __name__ == "__main__":
    main()
