import numpy as np
from scipy import linalg
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_fiedler_partition(A):
    N = A.shape[0]
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    try:
        vals, vecs = linalg.eigh(L)
        fiedler_vec = vecs[:, 1]
        clusters = (fiedler_vec > 0).astype(int)
        T = (clusters[:, None] == clusters[None, :]).astype(int)
        return T, clusters
    except Exception as e:
        return np.ones((N, N)), np.zeros(N)

def fcca(subject_matrices):
    N, _, S = subject_matrices.shape
    co_occurrence = np.zeros((N, N))
    for s in range(S):
        T_s, _ = get_fiedler_partition(subject_matrices[:, :, s])
        co_occurrence += T_s
    co_occurrence /= S
    _, consensus_clusters = get_fiedler_partition(co_occurrence)
    return consensus_clusters, co_occurrence

def run_analysis():
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    results_path = base_path / "outputs" / "ho_rlsl"
    out_path = base_path / "outputs" / "analysis"
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load electrodes for labels
    elec_df = pd.read_csv(base_path / "sub-001" / "eeg" / "sub-001_task-ERN_electrodes.tsv", sep='\t')
    node_names = elec_df['name'].tolist()[:30] # Keep only EEG
    
    # Load HO-RLSL results
    correct_L = np.load(results_path / "correct_low_rank.npy")
    incorrect_L = np.load(results_path / "incorrect_low_rank.npy")
    
    # Define Windows
    # index 128 is 0ms
    # ERN window: 0 to 150ms -> 128 to 147
    ern_start, ern_end = 128, 147
    
    # Average across ERN window
    correct_ern = np.mean(correct_L[:, :, ern_start:ern_end, :], axis=2) # (N, N, S)
    incorrect_ern = np.mean(incorrect_L[:, :, ern_start:ern_end, :], axis=2) # (N, N, S)
    
    # FCCA Clustering
    logger.info("Running FCCA for Correct trials...")
    clusters_c, W_c = fcca(correct_ern)
    logger.info("Running FCCA for Incorrect trials...")
    clusters_i, W_i = fcca(incorrect_ern)
    
    # Save clustering results
    results = pd.DataFrame({
        'Channel': node_names,
        'Cluster_Correct': clusters_c,
        'Cluster_Incorrect': clusters_i
    })
    results.to_csv(out_path / "clustering_results.csv", index=False)
    
    # Plot Connectivity Matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Mean across subjects for visualization
    im1 = axes[0].imshow(np.mean(correct_ern, axis=-1), cmap='jet', vmin=0, vmax=1)
    axes[0].set_title("Correct ERN Connectivity (Denoised)")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(np.mean(incorrect_ern, axis=-1), cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Incorrect ERN Connectivity (Denoised)")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(out_path / "ern_connectivity_comparison.png")
    logger.info(f"Comparison plot saved to {out_path / 'ern_connectivity_comparison.png'}")
    
    # Plot Co-occurrence Matrices (W)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    im1 = axes[0].imshow(W_c, cmap='hot', vmin=0, vmax=1)
    axes[0].set_title("Correct Co-occurrence (FCCA)")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(W_i, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title("Incorrect Co-occurrence (FCCA)")
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(out_path / "fcca_co_occurrence.png")
    logger.info(f"FCCA plots saved.")

if __name__ == "__main__":
    run_analysis()
