import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

class HORLSDecomposer:
    def __init__(self, n_nodes, n_subs, r=5, alpha_win=8, sigma_min=0.063, device='cpu'):
        self.N = n_nodes
        self.S = n_subs
        self.r = r  # Tucker rank
        self.alpha = alpha_win  # Window size for updates (Ozdemir used 8 for EEG)
        self.sigma_min_paper = sigma_min  # Precisely calibrated for 34 subjects, 30 channels
        self.device = device
        
        # Subspaces for Channel (U) and Subject (V)
        self.U = None 
        self.V = None

    def tucker_init(self, tensor_4d, n_init=128):
        """
        GIAI ĐOẠN HUẤN LUYỆN (Training Phase) - Theo Ozdemir (2017).
        Sử dụng toàn bộ Baseline (pre-stimulus) để tạo Subspace ổn định.
        """
        # Lấy đoạn Baseline làm dữ liệu huấn luyện
        m_train = tensor_4d[:, :, :, :n_init].to(self.device)
        
        # 1. Tìm Basis cho Mode Channel (N)
        # Unfold theo Channel: (N, S * N * n_init)
        m_chan = m_train.permute(1, 0, 2, 3).reshape(self.N, -1)
        u_chan, s_chan, _ = torch.svd(m_chan)
        self.U = u_chan[:, :self.r]
        
        # 2. Tính ngưỡng sigma_min = 10% của giá trị suy biến lớn nhất (Line 1097 bài báo)
        # Đây là mấu chốt để tránh Change-point giả ở Baseline
        self.sigma_min = 0.1 * s_chan[0].item()
        
        # 3. Tìm Basis cho Mode Subject (S)
        m_sub = m_train.reshape(self.S, -1)
        u_sub, _, _ = torch.svd(m_sub)
        self.V = u_sub[:, :self.r]
        
        print(f"  [Init] Subspace initialized on {n_init} samples (Baseline). Adaptive sigma_min: {self.sigma_min:.4f}")
        
    def update_subspace(self, data_window):
        """
        BƯỚC B: Add/Delete Directions (Trái tim của HO-RLSL).
        Sử dụng ngưỡng cố định sigma_min = 0.11 (Ozdemir 2017) cho dữ liệu EEG.
        """
        # 1. Chiếu dữ liệu lên phần bù vuông góc
        eye_n = torch.eye(self.N, device=self.device)
        proj_perp = eye_n - torch.matmul(self.U, self.U.T)
        
        flat_data = data_window.permute(1, 0, 2, 3).reshape(self.N, -1)
        w = flat_data.shape[1] # Số cột dữ liệu trong cửa sổ alpha
        projected_data = torch.matmul(proj_perp, flat_data)
        
        # 2. Tìm các hướng mới (Add Direction)
        u_new, s_new, _ = torch.svd(projected_data)
        
        # CHUYỂN ĐỔI NGƯỠNG: s^2 / w > sigma_min => s > sqrt(sigma_min * w)
        threshold = np.sqrt(self.sigma_min_paper * w)
        added_indices = torch.where(s_new > threshold)[0]
        
        if len(added_indices) > 0:
            new_dirs = u_new[:, added_indices]
            combined = torch.cat([self.U, new_dirs], dim=1)
            
            # 3. Chọn r thành phần mạnh nhất
            u_rot, s_rot, _ = torch.svd(torch.matmul(combined.T, flat_data))
            self.U = torch.matmul(combined, u_rot[:, :self.r])
            self.U, _ = torch.linalg.qr(self.U)
            
        return len(added_indices)

    def recover_sparse(self, X_t, max_iter=10, lambda_sparse=0.05):
        """
        Tách nhiễu thưa S_t bằng l1-minimization (ISTA).
        Tăng số vòng lặp và điều chỉnh lambda để khử artifact tốt hơn.
        """
        eye_n = torch.eye(self.N, device=self.device)
        phi_u = eye_n - torch.matmul(self.U, self.U.T)
        
        eye_s = torch.eye(self.S, device=self.device)
        phi_v = eye_s - torch.matmul(self.V, self.V.T)

        S_t = torch.zeros_like(X_t)
        
        for _ in range(max_iter):
            residual = X_t - S_t
            
            # Chiếu residual lên phần không thuộc Low-rank
            proj_res = torch.tensordot(residual, phi_u, dims=([1], [0]))
            proj_res = torch.tensordot(proj_res, phi_u, dims=([1], [0]))
            proj_res = torch.tensordot(proj_res, phi_v, dims=([0], [0]))
            proj_res = proj_res.permute(2, 0, 1)

            # ISTA Update
            S_t = S_t + proj_res
            S_t = torch.sign(S_t) * torch.clamp(torch.abs(S_t) - lambda_sparse, min=0)
            
        return S_t

    def decompose(self, tensor_4d):
        n_subs, n_chan, _, n_times = tensor_4d.shape
        
        # Tucker Initialization (Baseline - 1s pre-stimulus)
        self.tucker_init(tensor_4d, n_init=128)
        
        energy = torch.zeros(n_times, device=self.device)
        weights_evolution = torch.zeros((n_chan, n_times), device=self.device)
        subspace_velocity = torch.zeros(n_times, device=self.device)
        lowrank_tensor = torch.zeros_like(tensor_4d, device=self.device)
        lowrank_buffer = torch.zeros_like(tensor_4d, device=self.device)
        change_points = []
        
        U_prev = self.U.clone()
        
        for t in range(n_times):
            X_t = tensor_4d[:, :, :, t].to(self.device)
            
            # 1. Robust Recovery (Ozdemir Step 8)
            S_t = self.recover_sparse(X_t)
            L_clean_t = X_t - S_t
            lowrank_buffer[:, :, :, t] = L_clean_t
            
            # 2. Tucker Projection (Core Tensor Energy)
            temp = torch.tensordot(L_clean_t, self.U, dims=([1], [0]))
            temp = torch.tensordot(temp, self.U, dims=([1], [0]))
            core = torch.tensordot(temp, self.V, dims=([0], [0]))
            
            energy[t] = torch.norm(core)**2
            weights_evolution[:, t] = self.U[:, 0]
            
            # 3. Tính tốc độ xoay Subspace (Velocity)
            proj_current = torch.matmul(self.U, self.U.T)
            proj_prev = torch.matmul(U_prev, U_prev.T)
            velocity = torch.norm(proj_current - proj_prev)
            subspace_velocity[t] = velocity
            U_prev = self.U.clone()
            
            # 4. Tái cấu trúc Low-rank (Để visualize)
            l_temp = torch.tensordot(core, self.V, dims=([2], [1]))
            l_temp = torch.tensordot(l_temp, self.U, dims=([0], [1]))
            lowrank_tensor[:, :, :, t] = torch.tensordot(l_temp, self.U, dims=([0], [1]))
            
            # 5. Recursive Update (alpha = 8)
            # CHỈ cập nhật sau giai đoạn Baseline (n_init=128) để tránh Change-point giả
            if t >= 128 and t % self.alpha == 0: 
                window = lowrank_buffer[:, :, :, t-self.alpha : t]
                # n_new_dirs là số lượng hướng mới vượt ngưỡng sigma_min
                n_new_dirs = self.update_subspace(window)
                if n_new_dirs >= 1: # Theo bài báo: bất kỳ hướng mới nào cũng là Change Point
                    change_points.append(t)

        return weights_evolution.cpu().numpy(), energy.cpu().numpy(), subspace_velocity.cpu().numpy(), lowrank_tensor.cpu().numpy(), np.array(change_points, dtype=int)

def find_change_points(energy, velocity, threshold_factor=1.8):
    # Deprecated in favor of 'is_updated' flag in HORLSDecomposer.decompose
    return []

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Refining Phase 4 Analysis on {device} to match Ozdemir Paper...")
    
    # Process both conditions
    for cond, tensor_file in zip(['incorrect', 'correct'], [TENSOR_INCORRECT_FILE, TENSOR_CORRECT_FILE]):
        print(f"  Processing {cond} condition...")
        data = np.load(tensor_file)
        tensor = torch.tensor(data, dtype=torch.float32, device=device)
        n_subs, n_nodes = tensor.shape[0], tensor.shape[1]
        
        decomposer = HORLSDecomposer(n_nodes=n_nodes, n_subs=n_subs, device=device)
        w_t, energy, velocity, L_t, cp_indices = decomposer.decompose(tensor)
        
        # Save results for statistical validation
        np.save(TENSOR_DIR / f"horls_weights_{cond}.npy", w_t)
        np.save(TENSOR_DIR / f"horls_energy_{cond}.npy", energy)
        np.save(TENSOR_DIR / f"horls_lowrank_{cond}.npy", L_t)
        np.save(TENSOR_DIR / f"horls_cp_{cond}.npy", cp_indices)
        
        if cond == 'incorrect':
            time_ms = np.linspace(-1000, 1000, 256)
            cp_times = time_ms[cp_indices]
            
            # --- PLOTTING LIKE OZDEMIR ---
            fig = plt.figure(figsize=(18, 14))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, 2, 1])
            
            # 1. Network Energy and Subspace Velocity
            ax0 = fig.add_subplot(gs[0])
            ax0.plot(time_ms, energy, color='black', linewidth=2, label='Common Network Energy')
            ax0_twin = ax0.twinx()
            ax0_twin.plot(time_ms, velocity, color='blue', alpha=0.4, linestyle='--', label='Subspace Velocity')
            for cp in cp_times:
                ax0.axvline(cp, color='red', linestyle='--', alpha=0.8, label='Change-point' if cp == cp_times[0] else None)
            ax0.set_title("Network Dynamics & Detected Change-points")
            ax0.set_ylabel("Energy")
            ax0_twin.set_ylabel("Velocity")
            ax0.legend(loc='upper left')
            ax0_twin.legend(loc='upper right')
            ax0.grid(True, alpha=0.3)
            
            # 2. Heatmap of Weights (Channel x Time)
            ax1 = fig.add_subplot(gs[1])
            sample_epo_files = list(EPOCHS_DIR.glob("*.fif"))
            if not sample_epo_files:
                print("Lỗi: Không tìm thấy tệp epoch để lấy tên kênh.")
                return
            sample_epo = mne.read_epochs(sample_epo_files[0], preload=False, verbose=False)
            ch_names = sample_epo.ch_names
            
            # Robust Z-score: Dùng Median/IQR để tránh edge artifacts
            baseline_w = w_t[:, 10:100] # Bỏ qua 10 mẫu đầu tiên bị nhiễu khởi tạo
            w_med = np.median(baseline_w, axis=1, keepdims=True)
            w_std = np.std(baseline_w, axis=1, keepdims=True) + 1e-6
            z_w = (w_t - w_med) / w_std
            
            # Clipping để tránh bão hòa màu và làm rõ structure
            z_w = np.clip(z_w, -3, 3)
            
            im = ax1.imshow(z_w[:, 10:], aspect='auto', extent=[-920, 1000, n_nodes, 0], cmap='RdBu_r')
            ax1.set_yticks(np.arange(n_nodes) + 0.5)
            ax1.set_yticklabels(ch_names, fontsize=8)
            ax1.set_title("Relative Network Weights (Robust Z-scored, Offset Adjusted)")
            ax1.set_xlabel("Time (ms)")
            ax1.axvline(0, color='black', linestyle='-', linewidth=2)
            plt.colorbar(im, ax=ax1, label='Z-score Intensity')
            
            # 3. Topomap at peak ERN (around 50-70ms)
            ax2 = fig.add_subplot(gs[2])
            # Find max energy point in 25-100ms
            ern_mask = (time_ms >= 25) & (time_ms <= 100)
            peak_idx = np.where(ern_mask)[0][np.argmax(energy[ern_mask])]
            peak_time = time_ms[peak_idx]
            
            info = mne.create_info(ch_names, SFREQ, ch_types='eeg')
            info.set_montage(MONTAGE_NAME)
            
            # Vẽ Delta Weights (Sự thay đổi so với baseline)
            delta_w = w_t[:, peak_idx] - w_med.flatten()
            
            from mne.viz import plot_topomap
            im_topo, _ = plot_topomap(delta_w, info, axes=ax2, show=False, cmap='YlOrRd')
            ax2.set_title(f"Network Reconfiguration Topology at {peak_time:.1f} ms (Delta Weights)")
            
            plt.tight_layout()
            output_img = OUTPUTS_DIR / "ozdemir_replicated_dynamics.png"
            plt.savefig(output_img)
            print(f"Refined Paper-Style plot saved to: {output_img}")

if __name__ == "__main__":
    main()
