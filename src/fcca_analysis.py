import numpy as np
import mne
import pandas as pd
from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
from config import *

def run_fcca_analysis():
    print("Bắt đầu phân tích FCCA (Functional Canonical Correlation Analysis)...")
    
    # 1. Load Preprocessed Data (Refined Epochs)
    sub_files = list(REFINED_DIR.glob("*_theta_balanced-epo.fif"))
    sub_files.sort()
    
    # define modules
    modules = {
        'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz', 'F7', 'F8'],
        'F-Central': ['FC3', 'FC4', 'FCz'],
        'Central': ['Cz', 'C3', 'C4', 'C5', 'C6'],
        'Parietal': ['Pz', 'P3', 'P4', 'P7', 'P8', 'P9', 'P10', 'CPz'],
        'Occipital': ['POz', 'PO3', 'PO4', 'PO7', 'PO8', 'O1', 'O2', 'Oz']
    }
    
    module_names = list(modules.keys())
    n_mods = len(module_names)
    fcca_matrix_inc = np.zeros((n_mods, n_mods))
    fcca_matrix_cor = np.zeros((n_mods, n_mods))
    
    for i_sub, f in enumerate(sub_files):
        print(f"  Processing {f.name} ({i_sub+1}/{len(sub_files)})...")
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        ch_names = epochs.ch_names
        
        for cond, matrix in zip(['Incorrect', 'Correct'], [fcca_matrix_inc, fcca_matrix_cor]):
            data = epochs[cond].get_data() # (trials, channels, times)
            
            for i in range(n_mods):
                for j in range(i+1, n_mods):
                    # Get channels for each module
                    chs_i = [ch_names.index(ch) for ch in modules[module_names[i]] if ch in ch_names]
                    chs_j = [ch_names.index(ch) for ch in modules[module_names[j]] if ch in ch_names]
                    
                    if not chs_i or not chs_j: continue
                    
                    # Prepare X and Y (Combining trials and time for canonical correlation)
                    # X: (trials * times, channels_in_module_i)
                    X = data[:, chs_i, :].transpose(0, 2, 1).reshape(-1, len(chs_i))
                    Y = data[:, chs_j, :].transpose(0, 2, 1).reshape(-1, len(chs_j))
                    
                    # CCA
                    cca = CCA(n_components=1)
                    X_c, Y_c = cca.fit_transform(X, Y)
                    corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]
                    
                    matrix[i, j] += corr
                    matrix[j, i] += corr # Symmetric
                    
    # Average across subjects
    fcca_matrix_inc /= len(sub_files)
    fcca_matrix_cor /= len(sub_files)
    
    # 3. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, mat, title in zip(axes, [fcca_matrix_cor, fcca_matrix_inc], ['Correct', 'Incorrect (ERN)']):
        im = ax.imshow(mat, vmin=0, vmax=0.6, cmap='YlGnBu')
        ax.set_xticks(range(n_mods))
        ax.set_xticklabels(module_names, rotation=45)
        ax.set_yticks(range(n_mods))
        ax.set_yticklabels(module_names)
        ax.set_title(f"FCCA Module Interaction: {title}")
        plt.colorbar(im, ax=ax)
        
        # Add text values
        for r in range(n_mods):
            for c in range(n_mods):
                ax.text(c, r, f"{mat[r,c]:.2f}", ha='center', va='center', color='black')

    plt.tight_layout()
    output_img = OUTPUTS_DIR / "fcca_module_relationships.png"
    plt.savefig(output_img)
    print(f"FCCA Analysis complete. Results saved to: {output_img}")

if __name__ == "__main__":
    run_fcca_analysis()
