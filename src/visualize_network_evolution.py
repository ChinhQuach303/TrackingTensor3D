import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_connectivity.viz import plot_connectivity_circle
from config import *

def plot_network_evolution():
    # 1. Load Data
    w_inc = np.load(TENSOR_DIR / "horls_weights_incorrect.npy") # (30, 256)
    w_cor = np.load(TENSOR_DIR / "horls_weights_correct.npy")
    
    # Get channel names and positions
    sample_epo = mne.read_epochs(list(EPOCHS_DIR.glob("*.fif"))[0], preload=False, verbose=False)
    ch_names = sample_epo.ch_names
    
    # define groups for circular plot
    groups = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8'],
        'Fronto-Central': ['FC3', 'FC4', 'FCz'],
        'Central': ['Cz', 'C3', 'C4', 'C5', 'C6'],
        'Parietal': ['Pz', 'P3', 'P4', 'P7', 'P8', 'P9', 'P10', 'CPz'],
        'Occipital': ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Oz']
    }
    
    # Flat list of channels in group order for circular plot
    node_order = []
    group_boundaries = [0]
    for g_name, g_chs in groups.items():
        node_order.extend([ch for ch in g_chs if ch in ch_names])
        group_boundaries.append(len(node_order))
        
    node_idx = [ch_names.index(ch) for ch in node_order]
    
    # 2. Define Time Windows (Indices based on 256 points [-1s to 1s])
    # t=0 is at index 128
    windows = {
        'Pre-Response (-200 to 0ms)': (128 - 25, 128),
        'ERN/CRN Window (25 to 100ms)': (128 + 3, 128 + 13),
        'Post-Response (200 to 500ms)': (128 + 25, 128 + 64)
    }
    
    fig = plt.figure(figsize=(18, 12))
    
    for i, (win_name, (start, end)) in enumerate(windows.items()):
        # Compute mean interaction matrix for this window
        # Interaction between node i and j is proportional to w[i]*w[j] (Rank-1 property)
        
        for cond_idx, (w_t, cond_name) in enumerate([(w_inc, 'Incorrect'), (w_cor, 'Correct')]):
            avg_w = np.mean(w_t[:, start:end], axis=1) # (30,)
            
            # Reconstruct connectivity matrix from Rank-1 weights
            con_mat = np.outer(avg_w, avg_w)
            
            # Reorder for circular plot
            con_mat = con_mat[node_idx, :][:, node_idx]
            
            # Plot
            ax = fig.add_subplot(2, 3, i + 1 + (cond_idx * 3), projection='polar')
            
            plot_connectivity_circle(
                con_mat, node_order, n_lines=50,
                node_angles=None, title=f"{cond_name}\n{win_name}",
                ax=ax, colorbar=False, show=False,
                node_colors=None, padding=0, linewidth=2
            )
            
    plt.tight_layout()
    output_img = OUTPUTS_DIR / "network_evolution_circular.png"
    plt.savefig(output_img)
    print(f"Circular Network Evolution plot saved to: {output_img}")

if __name__ == "__main__":
    # Ensure mne-connectivity is installed for this specific visualization
    try:
        import mne_connectivity
    except ImportError:
        print("Installing mne-connectivity for advanced visualization...")
        import os
        os.system("pip install mne-connectivity")
        
    plot_network_evolution()
