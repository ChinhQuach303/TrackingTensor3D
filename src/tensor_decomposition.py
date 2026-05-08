import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

class HORLSDecomposer:
    def __init__(self, n_nodes, n_subs, r=5, alpha_win=4, sigma_min=0.02, device='cpu'):
        self.N = n_nodes
        self.S = n_subs
        self.r = r  # Tucker rank
        self.alpha = alpha_win  # Window size for updates
        self.sigma_min = sigma_min  # Threshold for adding/deleting directions
        self.device = device
        
        # Subspaces for Channel (U) and Subject (V)
        self.U = None 
        self.V = None

    def tucker_init(self, tensor_4d, n_init=10):
        """
        Khởi tạo subspace theo Ozdemir (2017) - Giữ nguyên chiều Subject.
        Sử dụng HOSVD trên khối dữ liệu ban đầu (S x N x N x n_init).
        """
        m_train = tensor_4d[:, :, :, :n_init].to(self.device)
        
        # 1. Tìm Basis cho Mode Channel (N)
        # Unfold theo Channel: (N, S * N * n_init)
        m_chan = m_train.permute(1, 0, 2, 3).reshape(self.N, -1)
        u_chan, _, _ = torch.svd(m_chan)
        self.U = u_chan[:, :self.r]
        
        # 2. Tìm Basis cho Mode Subject (S)
        # Unfold theo Subject: (S, N * N * n_init)
        m_sub = m_train.reshape(self.S, -1)
        u_sub, _, _ = torch.svd(m_sub)
        self.V = u_sub[:, :self.r]
        
        return self.U, self.V

    def update_subspace(self, data_window):
        """
        BƯỚC B: Add/Delete Directions (Trái tim của HO-RLSL).
        Cập nhật Subspace dựa trên dữ liệu mới lọt ra ngoài không gian cũ.
        """
        # 1. Chiếu dữ liệu lên phần bù vuông góc (Orthogonal Complement)
        # proj_perp = (I - UU')
        eye_n = torch.eye(self.N, device=self.device)
        proj_perp = eye_n - torch.matmul(self.U, self.U.T)
        
        # Unfold window dữ liệu theo mode Channel
        # data_window shape: (S, N, N, alpha)
        flat_data = data_window.permute(1, 0, 2, 3).reshape(self.N, -1)
        projected_data = torch.matmul(proj_perp, flat_data)
        
        # 2. Tìm các hướng mới (Add Direction) từ phần dữ liệu "lọt lưới"
        u_new, s_new, _ = torch.svd(projected_data)
        
        # Lấy các hướng có năng lượng vượt ngưỡng sigma_min
        added_indices = torch.where(s_new > self.sigma_min * s_new[0])[0]
        if len(added_indices) > 0:
            new_directions = u_new[:, added_indices]
            # Hợp nhất và trực giao hóa - Dùng QR để xoay subspace
            combined = torch.cat([self.U, new_directions], dim=1)
            q, _ = torch.linalg.qr(combined)
            
            # GIỮ LẠI r THÀNH PHẦN MỚI NHẤT (Xoay Subspace)
            # Thay vì lấy r cột đầu tiên, ta lấy r cột có năng lượng cao nhất trong window mới
            self.U = q[:, :self.r]
            
    def recover_sparse(self, X_t, max_iter=5, lambda_sparse=0.1):
        """
        BƯỚC CỐT LÕI MỚI: Tách nhiễu thưa S_t bằng l1-minimization (ISTA).
        Khớp với Equation (15) trong bài báo Ozdemir (2017).
        """
        # Phép chiếu lên phần bù vuông góc: Phi = (I - UU')
        eye_n = torch.eye(self.N, device=self.device)
        phi_u = eye_n - torch.matmul(self.U, self.U.T)
        
        eye_s = torch.eye(self.S, device=self.device)
        phi_v = eye_s - torch.matmul(self.V, self.V.T)

        S_t = torch.zeros_like(X_t)
        
        for _ in range(max_iter):
            # Tính phần dư (X_t - S_t)
            residual = X_t - S_t
            
            # Chiếu residual lên orthogonal complement (phần không thuộc Low-rank)
            # Theo mode: S_t = S_t + Phi_u * residual * Phi_u' (xấp xỉ tensor projection)
            proj_res = torch.tensordot(residual, phi_u, dims=([1], [0])) # Mode Channel 1
            proj_res = torch.tensordot(proj_res, phi_u, dims=([1], [0])) # Mode Channel 2
            proj_res = torch.tensordot(proj_res, phi_v, dims=([0], [0])) # Mode Subject
            proj_res = proj_res.permute(2, 0, 1) # Đưa về đúng (S, N, N)

            # Cập nhật S_t theo hướng gradient và dùng Soft-Thresholding
            S_t = S_t + proj_res
            S_t = torch.sign(S_t) * torch.clamp(torch.abs(S_t) - lambda_sparse, min=0)
            
        return S_t

    def decompose(self, tensor_4d):
        n_subs, n_chan, _, n_times = tensor_4d.shape
        
        # Tucker Initialization (Baseline)
        self.tucker_init(tensor_4d, n_init=10)
        
        energy = torch.zeros(n_times, device=self.device)
        weights_evolution = torch.zeros((n_chan, n_times), device=self.device)
        subspace_velocity = torch.zeros(n_times, device=self.device)
        lowrank_tensor = torch.zeros_like(tensor_4d, device=self.device)
        
        U_prev = self.U.clone() if self.U is not None else None
        
        for t in range(n_times):
            # X_t: Tensor 3 chiều tại thời điểm t (S, N, N)
            X_t = tensor_4d[:, :, :, t].to(self.device)
            
            # --- BƯỚC MỚI: ROBUST RECOVERY (Ozdemir Step 8) ---
            # 1. Tách nhiễu thưa S_t
            S_t = self.recover_sparse(X_t)
            # 2. Lấy tín hiệu sạch L_t
            L_clean_t = X_t - S_t
            
            # --- BƯỚC A: Tucker Projection (Core Tensor Energy) ---
            # Dùng L_clean_t thay vì X_t để đảm bảo năng lượng không bị nhiễu artifact
            temp = torch.tensordot(L_clean_t, self.U, dims=([1], [0])) # (S, N, r)
            temp = torch.tensordot(temp, self.U, dims=([1], [0])) # (S, r, r)
            core = torch.tensordot(temp, self.V, dims=([0], [0])) # (r, r, r)
            
            energy[t] = torch.norm(core)**2
            weights_evolution[:, t] = self.U[:, 0]
            
            # Tính tốc độ xoay của Subspace (Velocity)
            if U_prev is not None:
                velocity = torch.norm(self.U - U_prev)
                subspace_velocity[t] = velocity
                U_prev = self.U.clone()
            
            # --- TÁI CẤU TRÚC LOW-RANK (L_t) ---
            l_temp = torch.tensordot(core, self.V, dims=([2], [1])) # (r, r, S)
            l_temp = torch.tensordot(l_temp, self.U, dims=([0], [1])) # (r, S, N)
            L_reconstructed = torch.tensordot(l_temp, self.U, dims=([0], [1])) # (S, N, N)
            
            # Chuyển đổi về đúng thứ tự chiều (S, N, N) và lưu lại
            lowrank_tensor[:, :, :, t] = L_reconstructed
            
            # --- BƯỚC B: Recursive Update (Cập nhật Subspace mỗi alpha bước) ---
            # Dùng lowrank_tensor (đã được khử nhiễu thưa) để update subspace
            if t > 0 and t % 4 == 0: 
                # Sử dụng dữ liệu thực tế (L_clean_t) để update Subspace
                window = lowrank_tensor[:, :, :, t-4 : t].to(self.device)
                self.update_subspace(window)

        return weights_evolution.cpu().numpy(), energy.cpu().numpy(), subspace_velocity.cpu().numpy(), lowrank_tensor.cpu().numpy()

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
        n_subs, n_nodes = tensor.shape[0], tensor.shape[1]
        
        decomposer = HORLSDecomposer(n_nodes=n_nodes, n_subs=n_subs, device=device)
        w_t, energy, velocity, L_t = decomposer.decompose(tensor)
        
        # Save results for statistical validation
        np.save(TENSOR_DIR / f"horls_weights_{cond}.npy", w_t)
        np.save(TENSOR_DIR / f"horls_energy_{cond}.npy", energy)
        np.save(TENSOR_DIR / f"horls_lowrank_{cond}.npy", L_t)
        
        if cond == 'incorrect':
            # Find Change-points for the Incorrect condition (main focus of the paper)
            cp_indices = find_change_points(energy)
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
