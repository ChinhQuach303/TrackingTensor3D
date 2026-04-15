import mne
import numpy as np
import pandas as pd
from pathlib import Path
import os

base_path = Path('/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible')
subjects = ['sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005']
event_codes = [111, 212, 121, 222, 112, 221, 122, 211]

print(f'| Subject | Bad Trials (%) | Total Trials |')
print(f'|---------|---------------|--------------|')

for sub in subjects:
    try:
        set_file = base_path / sub / 'eeg' / f'{sub}_task-ERN_eeg.set'
        events_file = base_path / sub / 'eeg' / f'{sub}_task-ERN_events.tsv'
        
        # Load data
        raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
        raw.filter(1, 40, verbose=False)
        
        # Process events
        events_df = pd.read_csv(events_file, sep='\t')
        mne_events = []
        for _, row in events_df.iterrows():
            if int(row['value']) in event_codes:
                sample = int(round(row['onset'] * raw.info['sfreq']))
                mne_events.append([sample, 0, int(row['value'])])
        mne_events = np.array(mne_events)
        
        # Epoching
        epochs = mne.Epochs(raw, mne_events, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, verbose=False)
        
        # Check ptp
        data = epochs.get_data(picks='eeg') * 1e6
        ptp = np.max(data, axis=-1) - np.min(data, axis=-1)
        bad_rate = np.mean(np.any(ptp > 100, axis=1)) * 100
        
        print(f'| {sub} | {bad_rate:.2f}% | {len(epochs)} |')
    except Exception as e:
        print(f'| {sub} | ERROR: {str(e)} | - |')
