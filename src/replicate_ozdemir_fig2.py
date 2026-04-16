import numpy as np
import matplotlib.pyplot as plt
import mne
from pathlib import Path
from config import *

def plot_replicated_fig2():
    print("Bắt đầu tái hiện Figure 2 (Ozdemir Style)...")
    
    # 1. Load ERP Data for FCz
    sub_files = list(EPOCHS_DIR.glob("*.fif"))
    all_erp_inc = []
    all_erp_cor = []
    
    for f in sub_files:
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        if 'FCz' in epochs.ch_names:
            all_erp_inc.append(epochs['Incorrect'].average().pick_channels(['FCz']).data[0])
            all_erp_cor.append(epochs['Correct'].average().pick_channels(['FCz']).data[0])
            
    ga_inc = np.mean(all_erp_inc, axis=0) * 1e6 # to microvolts
    ga_cor = np.mean(all_erp_cor, axis=0) * 1e6
    time_ms = np.linspace(-1000, 1000, ga_inc.shape[0])
    
    # 2. Load Change-points and Weights
    cp_inc_raw = np.load(TENSOR_DIR / "horls_cp_incorrect.npy")
    cp_cor_raw = np.load(TENSOR_DIR / "horls_cp_correct.npy")
    w_inc = np.load(TENSOR_DIR / "horls_weights_incorrect.npy")
    
    # Define threshold for CP detection (same as before)
    def get_cp_times(score, times):
        diff = np.abs(np.diff(score))
        th = np.mean(diff) + 1.5 * np.std(diff)
        idx = np.where(diff > th)[0]
        distinct = [idx[0]] if len(idx) > 0 else []
        for i in idx[1:]:
            if i - distinct[-1] > 20: distinct.append(i)
        return times[distinct]

    cp_inc_times = get_cp_times(cp_inc_raw, np.linspace(-1000, 1000, 256))
    cp_cor_times = get_cp_times(cp_cor_raw, np.linspace(-1000, 1000, 256))

    # 3. SETUP FIGURE
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1])
    
    # (a) Incorrect Waveform + Change-points
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(time_ms, ga_inc, color='red', linewidth=1.5)
    for cp in cp_inc_times:
        ax_a.axvline(cp, color='blue', linestyle='-', alpha=0.6)
        ax_a.text(cp, np.max(ga_inc), f"{cp:.0f}ms", rotation=0, ha='center', fontsize=9)
    ax_a.set_title("(a) Grand Average ERN at FCz & Change-points")
    ax_a.set_ylabel("Amplitude (μV)")
    ax_a.invert_yaxis() # EEG convention: negative up? Ozdemir image shows negative down. 
    # Let's keep negative down like their plot.
    ax_a.grid(True, alpha=0.3)
    
    # (b) Correct Waveform + Change-points
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.plot(time_ms, ga_cor, color='red', linewidth=1.5)
    for cp in cp_cor_times:
        ax_b.axvline(cp, color='blue', linestyle='-', alpha=0.6)
    ax_b.set_title("(b) Grand Average CRN at FCz & Change-points")
    ax_b.set_ylabel("Amplitude (μV)")
    ax_b.set_xlabel("Time (ms)")
    ax_b.grid(True, alpha=0.3)

    # (c) Topological Networks
    # We will pick the peak ERN window for topological connectivity
    info = mne.create_info(all_chs := epochs.ch_names, SFREQ, ch_types='eeg')
    info.set_montage(MONTAGE_NAME)
    pos = info.get_montage().get_positions()['ch_pos']
    xy = np.array([[pos[ch][0], pos[ch][1]] for ch in all_chs])
    
    ax_c = fig.add_subplot(gs[:, 1])
    # Draw head
    mask = np.ones(len(all_chs), dtype=bool)
    from mne.viz import plot_topomap
    plot_topomap(np.zeros(len(all_chs)), info, axes=ax_c, show=False, contours=0, sphere=0.1, sensors=True)
    
    # Draw edges for Incorrect peak
    # Interaction = w_i * w_j
    peak_idx = 128 + 10 # around 80ms
    current_w = w_inc[:, peak_idx]
    adj = np.outer(current_w, current_w)
    
    # Plot top 50 strongest connections
    flat_adj = adj.flatten()
    indices = np.argsort(flat_adj)[-100:] # Top 50 pairs
    
    for idx in indices:
        row, col = divmod(idx, len(all_chs))
        if row >= col: continue
        # Normalize weight for visibility
        p1, p2 = xy[row][:2], xy[col][:2]
        ax_c.plot([p1[0], p2[0]], [p1[1], p2[1]], color='cyan', alpha=0.4, linewidth=1)

    # Draw nodes
    ax_c.scatter(xy[:, 0], xy[:, 1], c='orange', s=50, edgecolors='black', zorder=10)
    for i, txt in enumerate(all_chs):
        ax_c.annotate(txt, (xy[i, 0], xy[i, 1]), fontsize=8, fontweight='bold')
        
    ax_c.set_title("(c) Identified Network Topology during ERN")
    
    plt.tight_layout()
    output_img = OUTPUTS_DIR / "ozdemir_fig2_replication.png"
    plt.savefig(output_img)
    print(f"Fig 2 Replication saved to: {output_img}")

if __name__ == "__main__":
    plot_replicated_fig2()
