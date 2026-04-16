import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

def plot_ozdemir_evolution_network():
    print("Đang tạo bản đồ tiến hóa mạng lưới (Bản chuẩn Ozdemir Fig 2 Right)...")
    
    # 1. Load Data
    w_inc = np.load(TENSOR_DIR / "horls_weights_incorrect.npy") # (30, 256)
    
    sample_epo = mne.read_epochs(list(EPOCHS_DIR.glob("*.fif"))[0], preload=False, verbose=False)
    ch_names = sample_epo.ch_names
    info = mne.create_info(ch_names, SFREQ, ch_types='eeg')
    info.set_montage(MONTAGE_NAME)
    pos = info.get_montage().get_positions()['ch_pos']
    xy = np.array([[pos[ch][0], pos[ch][1]] for ch in ch_names])

    # Define color-coded modules
    modules = {
        'Frontal': (['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8'], 'cyan'),
        'Central': (['FC3', 'FC4', 'FCz', 'Cz', 'C3', 'C4', 'C5', 'C6'], 'magenta'),
        'Parietal-Occipital': (['Pz', 'P3', 'P4', 'P7', 'P8', 'P9', 'P10', 'O1', 'O2', 'Oz', 'POz', 'CPz'], 'purple')
    }
    
    # Map channel to color
    ch_to_color = {}
    for mod, (chs, color) in modules.items():
        for ch in chs:
            if ch in ch_names: ch_to_color[ch] = color

    # 2. Define 3 Key Time Points (based on our Change-points)
    # t=0 is at 128. 
    times_idx = {
        'PRE-ERN (-200ms)': 128 - 25,
        'ERN (Peak 60ms)': 128 + 8,
        'POST-ERN (300ms)': 128 + 38
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    
    for i, (name, t_idx) in enumerate(times_idx.items()):
        ax = axes[i]
        
        # Draw Head Outline
        mne.viz.plot_topomap(np.zeros(len(ch_names)), info, axes=ax, show=False, 
                             contours=0, sphere=0.1, sensors=True)
        
        # Compute Connectivity Matrix for this sample
        weights = w_inc[:, t_idx]
        adj = np.outer(weights, weights)
        
        # Get top edges
        flat_adj = adj.flatten()
        top_indices = np.argsort(flat_adj)[-150:] # Top connections
        
        for idx in top_indices:
            r, c = divmod(idx, len(ch_names))
            if r >= c: continue
            
            p1, p2 = xy[r][:2], xy[c][:2]
            
            # Color logic based on paper: 
            # If within same module -> use module color. If cross -> use orange/gray.
            c1, c2 = ch_names[r], ch_names[c]
            color1 = ch_to_color.get(c1, 'gray')
            color2 = ch_to_color.get(c2, 'gray')
            
            edge_color = color1 if color1 == color2 else 'orange'
            linewidth = weights[r] * weights[c] * 100 # scale by interaction strength
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=edge_color, alpha=0.5, linewidth=linewidth)

        # Plot nodes with labels
        ax.scatter(xy[:, 0], xy[:, 1], c='white', s=80, edgecolors='black', zorder=10)
        for j, txt in enumerate(ch_names):
            ax.annotate(txt, (xy[j, 0], xy[j, 1]), fontsize=7, fontweight='bold', ha='center')
            
        ax.set_title(name, fontsize=15, fontweight='bold')

    plt.tight_layout()
    output_img = OUTPUTS_DIR / "ozdemir_network_evolution_topo.png"
    plt.savefig(output_img)
    print(f"Evolution Network Topo saved to: {output_img}")

if __name__ == "__main__":
    plot_ozdemir_evolution_network()
