import mne
import pandas as pd
import numpy as np
from pathlib import Path
import os

def refine_master_epochs(min_incorrect=15):
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    report_path = base_path / "data" / "processed" / "master_preprocessing_report.csv"
    epochs_dir = base_path / "data" / "processed" / "master_epochs"
    refined_dir = base_path / "data" / "processed" / "refined_master"
    refined_dir.mkdir(parents=True, exist_ok=True)
    
    if not report_path.exists():
        print("Error: master_preprocessing_report.csv not found.")
        return

    report = pd.read_csv(report_path)
    # Filter valid subjects
    valid_subs = report[(report['status'] == 'SUCCESS') & (report['incorrect'] >= min_incorrect)]
    
    print(f"Refining {len(valid_subs)} subjects (min incorrect: {min_incorrect})...")
    
    summary_data = []

    for _, row in valid_subs.iterrows():
        sub = row['sub']
        try:
            epochs = mne.read_epochs(epochs_dir / f"{sub}_master-epo.fif", preload=True, verbose=False)
            
            # 1. Trial Balancing (Undersample Correct)
            n_inc = len(epochs['Incorrect'])
            
            # Identify indices for Correct and Incorrect
            event_id_rev = {v: k for k, v in epochs.event_id.items()}
            correct_indices = [i for i, event in enumerate(epochs.events) if event_id_rev[event[2]].startswith('Correct')]
            incorrect_indices = [i for i, event in enumerate(epochs.events) if event_id_rev[event[2]].startswith('Incorrect')]
            
            # Randomly select n_inc from correct
            np.random.seed(42)
            balanced_correct = np.random.choice(correct_indices, size=n_inc, replace=False)
            
            # Combine and sort to maintain temporal order if needed
            final_indices = np.concatenate([balanced_correct, incorrect_indices])
            final_indices.sort()
            
            epochs_balanced = epochs[final_indices]
            
            # 2. Narrow-band Filter (4-8 Hz)
            # This is standard for ERN theta analysis
            epochs_theta = epochs_balanced.copy().filter(4, 8, fir_design='firwin', verbose=False)
            
            # Save
            out_file = refined_dir / f"{sub}_theta_balanced-epo.fif"
            epochs_theta.save(out_file, overwrite=True, verbose=False)
            
            summary_data.append({'sub': sub, 'final_trials_per_cond': n_inc})
            print(f"  DONE: {sub} | Balanced at {n_inc} trials.")
            
        except Exception as e:
            print(f"  FAILED: {sub} | {e}")

    # Save summary
    pd.DataFrame(summary_data).to_csv(base_path / "data" / "processed" / "refinement_summary.csv", index=False)
    print("-" * 50)
    print(f"Refinement complete. Valid subjects: {len(summary_data)}")

if __name__ == "__main__":
    # Threshold 15 is standard for stable PLV
    refine_master_epochs(min_incorrect=15)
