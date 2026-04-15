import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from scipy import linalg
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HORLSL:
    def __init__(self, ranks=(5, 5, 5), alpha=20, lmbda=0.01):
        """
        ranks: Tucker rank (r1, r2, r3)
        alpha: update window size
        lmbda: sparsity threshold
        """
        self.ranks = ranks
        self.alpha = alpha
        self.lmbda = lmbda
        self.bases = [None, None, None]
        self.low_rank_buffer = []  # Buffer for current subspace estimation
        self.change_points = []
        self.L_history = []
        self.S_history = []

    def initialize(self, M_init):
        """
        Initialize subspace using Tucker decomposition on a training set.
        M_init: (N1, N2, S, T_train)
        """
        logger.info(f"Initializing subspace with shape {M_init.shape}...")
        # Average across time or stack? Usually stack for Tucker
        # But memory-wise, averaging is safer if T_train is large.
        # Let's use the mean for initial subspace.
        M_mean = np.mean(M_init, axis=-1)
        core, factors = tl.decomposition.tucker(M_mean, rank=self.ranks, init='svd')
        self.bases = factors
        logger.info(f"Initial ranks: {self.ranks}")

    def soft_threshold(self, x, lmbda):
        return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

    def step(self, M_t, t_idx):
        """
        Process a single time point tensor M_t: (N1, N2, S)
        """
        # 1. Form projection operators
        phis = [np.eye(b.shape[0]) - b @ b.T for b in self.bases]
        
        # 2. Project measurement
        # Y_t = M_t x1 phi1 x2 phi2 x3 phi3
        Y_t = tl.tenalg.multi_mode_dot(M_t, phis)
        
        # 3. Sparse recovery (Simplified GTCS-S approx: soft thresholding)
        # In reality, S_t solves a more complex problem, but this is a common robust proxy.
        S_t = self.soft_threshold(Y_t, self.lmbda)
        
        # 4. Low-rank estimate
        L_t = M_t - S_t
        
        self.L_history.append(L_t)
        self.S_history.append(S_t)
        self.low_rank_buffer.append(L_t)
        
        # 5. Periodically update subspace
        if len(self.low_rank_buffer) >= self.alpha:
            self.update_subspace(t_idx)
            self.low_rank_buffer = []
            
        return L_t, S_t

    def update_subspace(self, t_idx):
        # Update each basis matrix via SVD on the buffer
        # buffer is list of (N1, N2, S) -> shape (N1, N2, S, alpha)
        D = np.stack(self.low_rank_buffer, axis=-1)
        
        subspace_changed = False
        for i in range(3):
            # Unfold and SVD
            unfolded = tl.unfold(D, i)
            U, s, Vh = linalg.svd(unfolded, full_matrices=False)
            new_basis = U[:, :self.ranks[i]]
            
            # Check for significant change (simplified change point detection)
            # Distance between subspaces: || (I - P_old P_old.T) P_new ||
            dist = np.linalg.norm((np.eye(new_basis.shape[0]) - self.bases[i] @ self.bases[i].T) @ new_basis)
            if dist > 0.5: # Threshold for change point
                subspace_changed = True
            
            self.bases[i] = new_basis
            
        if subspace_changed:
            self.change_points.append(t_idx)
            logger.info(f"Change point detected at t={t_idx}")

def run_ho_rlsl(tensor_path, output_path, name):
    logger.info(f"Loading 4D tensor: {tensor_path}")
    M = np.load(tensor_path)
    N1, N2, T, S = M.shape
    # Correct shape to (N1, N2, S, T) for processing
    M = np.moveaxis(M, 2, 3) # (30, 30, 40, 257)
    
    # 1. Initialization (using baseline: -1.0 to -0.2s)
    # -1.0 to 1.0 is 257 samples. 0 index is -1.0s. 
    # -0.2s is at index (0.8 * 128) = 102? No.
    # t = -1.0 + k/128. -0.2 = -1.0 + k/128 => 0.8 = k/128 => k = 102.4. So index 102.
    baseline_idx = 102
    horlsl = HORLSL(ranks=(5, 5, 5), alpha=20, lmbda=0.05)
    horlsl.initialize(M[:, :, :, :baseline_idx])
    
    # 2. Tracking
    logger.info("Starting recursive tracking...")
    for t in range(T):
        horlsl.step(M[:, :, :, t], t)
        if (t + 1) % 50 == 0:
            logger.info(f"  Processed {t+1}/{T} steps")
            
    # 3. Save results
    L = np.stack(horlsl.L_history, axis=-1)
    # Move back to (N, N, T, S)
    L = np.moveaxis(L, 3, 2)
    
    res_path = output_path / f"{name}_low_rank.npy"
    np.save(res_path, L)
    logger.info(f"Low-rank tensor saved to {res_path}")
    
    # Save change points
    cp_path = output_path / f"{name}_change_points.txt"
    with open(cp_path, 'w') as f:
        f.write('\n'.join(map(str, horlsl.change_points)))
    logger.info(f"Change points saved to {cp_path}")

if __name__ == "__main__":
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    tensor_path = base_path / "outputs" / "tensors"
    output_path = base_path / "outputs" / "ho_rlsl"
    output_path.mkdir(parents=True, exist_ok=True)
    
    run_ho_rlsl(tensor_path / "correct_4d_tensor.npy", output_path, "correct")
    run_ho_rlsl(tensor_path / "incorrect_4d_tensor.npy", output_path, "incorrect")
