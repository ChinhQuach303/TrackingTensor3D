import mne
import numpy as np
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA, compute_current_source_density
import os
from config import *

def preprocess_subject_ozdemir_style(sub_id):
    raw_path = BASE_PATH / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    
    # 1. Load Raw
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    
    # Rename for MNE Montage awareness (FP1 -> Fp1, etc.)
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw.rename_channels(lambda x: rename_map.get(x, x))
    
    # Set types
    raw.set_channel_types({ch: 'eog' for ch in EOG_CHANNELS if ch in raw.ch_names})
    raw.set_montage(MONTAGE_NAME, on_missing='ignore')
    
    # 2. Resampling & Filtering
    raw.resample(SFREQ)
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design='firwin', verbose=False)
    
    # 3. Pick Channels
    available_eeg = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    active_eog = [ch for ch in EOG_CHANNELS if ch in raw.ch_names]
    raw.pick_channels(available_eeg + active_eog)
    
    # 4. ICA
    ica = ICA(n_components=ICA_COMPONENTS, random_state=42, method=ICA_METHOD, fit_params=dict(extended=True))
    ica.fit(raw, verbose=False)
    # EOG Rejection
    eog_indices, _ = ica.find_bads_eog(raw, ch_name='Fp1', verbose=False)
    ica.exclude = eog_indices
    raw_cleaned = ica.apply(raw.copy(), verbose=False)
    
    # 5. CSD
    raw_eeg = raw_cleaned.copy().pick_types(eeg=True)
    valid_eeg = [ch for ch in raw_eeg.ch_names if raw_eeg.info['chs'][raw_eeg.ch_names.index(ch)]['loc'][:3].any()]
    raw_eeg.pick_channels(valid_eeg)
    raw_csd = compute_current_source_density(raw_eeg, verbose=False)
    
    # 6. Epoching
    events, event_id = mne.events_from_annotations(raw_csd, verbose=False)
    correct_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] == l[2]]
    incorrect_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] != l[2]]
    
    resp_id = {label: event_id[label] for label in correct_codes + incorrect_codes}
    epochs = mne.Epochs(raw_csd, events, event_id=resp_id, tmin=TMIN, tmax=TMAX, 
                        baseline=BASELINE, preload=True, detrend=1, verbose=False)
    
    mapping = {l: f"Correct/{l}" for l in correct_codes}
    mapping.update({l: f"Incorrect/{l}" for l in incorrect_codes})
    epochs.event_id = {mapping[k]: v for k, v in epochs.event_id.items()}
    
    save_file = EPOCHS_DIR / f"{sub_id}_master-epo.fif"
    epochs.save(save_file, overwrite=True, verbose=False)
    return epochs

if __name__ == "__main__":
    sub_dirs = [d.name for d in BASE_PATH.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    sub_dirs.sort()
    
    report_data = []
    print(f"Running Preprocessing using Global Config...")
    for sub in sub_dirs:
        try:
            ep = preprocess_subject_ozdemir_style(sub)
            report_data.append({'sub': sub, 'status': 'SUCCESS', 'correct': len(ep['Correct']), 'incorrect': len(ep['Incorrect'])})
            print(f"  DONE: {sub} ({len(ep.ch_names)} EEG channels)")
        except Exception as e:
            report_data.append({'sub': sub, 'status': f'FAILED: {e}', 'correct': 0, 'incorrect': 0})
            print(f"  FAILED: {sub} | {e}")
            
    pd.DataFrame(report_data).to_csv(REPORT_ETL, index=False)
