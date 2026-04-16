import numpy as np
import matplotlib.pyplot as plt
import mne
from config import *

def plot_modular_grand_average_30ch():
    # 1. Full 30 channels in spatial order
    modular_order = [
        'Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8', # Frontal
        'FC3', 'FC4', 'FCz',                       # Fronto-Central
        'Cz', 'C3', 'C4', 'C5', 'C6',              # Central
        'CPz',                                     # Centro-Parietal
        'Pz', 'P3', 'P4', 'P7', 'P8', 'P9', 'P10', # Parietal
        'POz', 'PO3', 'PO4', 'PO7', 'PO8',         # Parieto-Occipital
        'O1', 'O2', 'Oz'                           # Occipital
    ]
    
    # Get actual names
    sample_epo = mne.read_epochs(list(EPOCHS_DIR.glob("*.fif"))[0], preload=False, verbose=False)
    actual_chs = sample_epo.ch_names
    
    final_order = [ch for ch in modular_order if ch in actual_chs]
    reorder_idx = [actual_chs.index(ch) for ch in final_order]
    
    # 2. Load Tensors
    correct = np.load(TENSOR_CORRECT_FILE)
    incorrect = np.load(TENSOR_INCORRECT_FILE)
    
    # 3. Reorder axes 1 and 2
    correct = correct[:, reorder_idx, :, :][:, :, reorder_idx, :]
    incorrect = incorrect[:, reorder_idx, :, :][:, :, reorder_idx, :]
    
    # 4. Window 25-75ms (at 128Hz, t=0 is index 128. 25ms is ~3.2 samples)
    t_start = 128 + 3
    t_end = 128 + 10
    
    ga_correct = np.mean(correct[:, :, :, t_start:t_end], axis=(0, 3))
    ga_incorrect = np.mean(incorrect[:, :, :, t_start:t_end], axis=(0, 3))
    
    # 5. Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    im1 = axes[0].imshow(ga_correct, vmin=0, vmax=0.7, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f"30-CH Grand Average: Correct (25-75ms)")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(ga_incorrect, vmin=0, vmax=0.7, cmap='magma', interpolation='nearest')
    axes[1].set_title(f"30-CH Grand Average: Incorrect (ERN) (25-75ms)")
    plt.colorbar(im2, ax=axes[1])
    
    for ax in axes:
        ax.set_xticks(range(len(final_order)))
        ax.set_xticklabels(final_order, rotation=90, fontsize=8)
        ax.set_yticks(range(len(final_order)))
        ax.set_yticklabels(final_order, fontsize=8)
        
    plt.tight_layout()
    output_img = OUTPUTS_DIR / "modular_grand_average_30ch.png"
    plt.savefig(output_img)
    print(f"30-Channel Modular Grand Average saved to: {output_img}")

if __name__ == "__main__":
    plot_modular_grand_average_30ch()
