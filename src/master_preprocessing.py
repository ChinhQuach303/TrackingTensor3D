import mne
import numpy as np
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA, compute_current_source_density
import os

def preprocess_subject_ozdemir_style(sub_id):
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    raw_path = base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    output_path = base_path / "data" / "processed" / "master_epochs"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Raw
    raw = mne.io.read_raw_eeglab(raw_path, preload=True)
    
    # Rename for MNE Montage awareness (FP1 -> Fp1, etc.)
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw.rename_channels(lambda x: rename_map.get(x, x))
    
    # Identify EEG channels (those with type 'eeg' or matched via tsv)
    # The TSV says these are EEG:
    target_eeg = [
        'Fp1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz',
        'Fp2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2'
    ]
    
    active_eog = ['HEOG_left', 'HEOG_right', 'VEOG_lower']
    raw.set_channel_types({ch: 'eog' for ch in active_eog if ch in raw.ch_names})
    
    # IMPORTANT: Use standard_1005 for high density
    raw.set_montage("standard_1005", on_missing='ignore')
    
    # Check for missing locations
    missing_locs = [ch for ch in target_eeg if ch in raw.ch_names and not raw.info['chs'][raw.ch_names.index(ch)]['loc'].any()]
    if missing_locs:
        print(f"  WARNING: Missing locations for {missing_locs}. CSD may fail.")
        # If still missing, we might need a custom montage or fix labels
    
    # 2. Resampling & Filtering
    raw.resample(128)
    raw.filter(0.1, 30.0, fir_design='firwin')
    
    # 3. Pick all 30 EEG + EOG
    available_eeg = [ch for ch in target_eeg if ch in raw.ch_names]
    raw.pick_channels(available_eeg + active_eog)
    
    # 4. ICA
    ica = ICA(n_components=10, random_state=42, method='infomax', fit_params=dict(extended=True))
    ica.fit(raw)
    # Use Fp1 for EOG correlation
    eog_indices, _ = ica.find_bads_eog(raw, ch_name='Fp1', verbose=False)
    ica.exclude = eog_indices
    raw_cleaned = ica.apply(raw.copy())
    
    # 5. CSD (Only on EEG channels with locations)
    # Filter to only those with valid positions
    raw_eeg = raw_cleaned.copy().pick_types(eeg=True)
    valid_eeg = [ch for ch in raw_eeg.ch_names if raw_eeg.info['chs'][raw_eeg.ch_names.index(ch)]['loc'][:3].any()]
    raw_eeg.pick_channels(valid_eeg)
    
    raw_csd = compute_current_source_density(raw_eeg)
    
    # 6. Epoching Correct/Incorrect
    events, event_id = mne.events_from_annotations(raw_csd)
    correct_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] == l[2]]
    incorrect_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] != l[2]]
    
    resp_id = {label: event_id[label] for label in correct_codes + incorrect_codes}
    epochs = mne.Epochs(raw_csd, events, event_id=resp_id, tmin=-1.0, tmax=1.0, 
                        baseline=(-0.6, -0.4), preload=True, detrend=1)
    
    mapping = {l: f"Correct/{l}" for l in correct_codes}
    mapping.update({l: f"Incorrect/{l}" for l in incorrect_codes})
    epochs.event_id = {mapping[k]: v for k, v in epochs.event_id.items()}
    
    epochs.save(output_path / f"{sub_id}_master-epo.fif", overwrite=True)
    return epochs

if __name__ == "__main__":
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    sub_dirs = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    sub_dirs.sort()
    
    report_data = []
    print(f"Rerunning Bulk Preprocessing with CORRECT 30 EEG Channels...")
    for sub in sub_dirs:
        try:
            ep = preprocess_subject_ozdemir_style(sub)
            report_data.append({'sub': sub, 'status': 'SUCCESS', 'correct': len(ep['Correct']), 'incorrect': len(ep['Incorrect'])})
            print(f"DONE: {sub} ({len(ep.ch_names)} EEG channels)")
        except Exception as e:
            report_data.append({'sub': sub, 'status': f'FAILED: {e}', 'correct': 0, 'incorrect': 0})
            print(f"FAILED: {sub} | {e}")
            
    pd.DataFrame(report_data).to_csv(base_path / "data" / "processed" / "master_preprocessing_report.csv", index=False)
