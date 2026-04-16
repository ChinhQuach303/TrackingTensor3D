import mne
import numpy as np
from pathlib import Path
from config import *

def refine_dataset():
    sub_files = list(EPOCHS_DIR.glob("*_master-epo.fif"))
    sub_files.sort()
    
    print(f"Refining subjects (min incorrect: {MIN_INCORRECT_TRIALS})...")
    
    valid_count = 0
    for f in sub_files:
        sub_id = f.stem.split('_')[0]
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        
        n_correct = len(epochs['Correct'])
        n_incorrect = len(epochs['Incorrect'])
        
        if n_incorrect < MIN_INCORRECT_TRIALS:
            # print(f"  SKIP: {sub_id} (Too few incorrect: {n_incorrect})")
            continue
            
        # 1. Theta Filtering (4-8 Hz)
        epochs_theta = epochs.copy().filter(l_freq=THETA_BAND[0], h_freq=THETA_BAND[1], 
                                            fir_design='firwin', verbose=False)
        
        # 2. Trial Balancing (1:1 Ratio)
        n_match = min(n_correct, n_incorrect)
        
        # Random pick for balancing
        idx_correct = np.random.choice(len(epochs_theta['Correct']), n_match, replace=False)
        idx_incorrect = np.random.choice(len(epochs_theta['Incorrect']), n_match, replace=False)
        
        balanced_epochs = mne.concatenate_epochs([
            epochs_theta['Correct'][idx_correct],
            epochs_theta['Incorrect'][idx_incorrect]
        ], verbose=False)
        
        save_file = REFINED_DIR / f"{sub_id}_theta_balanced-epo.fif"
        balanced_epochs.save(save_file, overwrite=True, verbose=False)
        # print(f"  DONE: {sub_id} | Balanced at {n_match} trials.")
        valid_count += 1
        
    print("-" * 50)
    print(f"Refinement complete. Valid subjects: {valid_count}")

if __name__ == "__main__":
    refine_dataset()
