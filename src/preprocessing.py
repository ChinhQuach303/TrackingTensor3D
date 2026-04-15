import mne
import numpy as np
import os
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_subject(subject_id, data_path, output_path):
    try:
        logger.info(f"Processing {subject_id}...")
        eeg_dir = data_path / subject_id / 'eeg'
        set_files = list(eeg_dir.glob('*.set'))
        if not set_files:
            logger.warning(f"No .set file found for {subject_id}")
            return
        
        set_file = set_files[0]
        events_file = list(eeg_dir.glob(f'{subject_id}_task-ERN_events.tsv'))[0]
        
        # Load raw data
        # Note: eeglab files often come with .fdt. MNE handles this automatically if in same dir.
        raw = mne.io.read_raw_eeglab(str(set_file), preload=True)
        
        # Select EEG channels (exclude EOG or others if needed)
        # Based on channels.tsv, we have 31-33 channels.
        # We'll keep all for now, as PLV can be computed between subsets later.
        
        # Band-pass filter (4-8 Hz - Theta)
        raw.filter(l_freq=4.0, h_freq=8.0, fir_design='firwin', verbose=False)
        
        # Resample to 128 Hz
        raw.resample(sfreq=128, verbose=False)
        
        # Load events from tsv
        events_df = pd.read_csv(events_file, sep='\t')
        
        # MNE events format: [sample, 0, event_id]
        # Map 'onset' (seconds) to samples
        mne_events = []
        for _, row in events_df.iterrows():
            sample = int(round(row['onset'] * raw.info['sfreq']))
            mne_events.append([sample, 0, int(row['value'])])
        mne_events = np.array(mne_events)
        
        # Define response codes (Correct/Incorrect)
        correct_codes = [111, 212, 121, 222]
        incorrect_codes = [211, 112, 221, 122]
        
        event_dict = {
            'Correct': 1,
            'Incorrect': 0
        }
        
        # Remap events
        new_events = []
        for event in mne_events:
            val = event[2]
            if val in correct_codes:
                new_events.append([event[0], 0, 1])
            elif val in incorrect_codes:
                new_events.append([event[0], 0, 0])
        new_events = np.array(new_events)
        
        if len(new_events) == 0:
            logger.warning(f"No response events found for {subject_id}")
            return

        # Define epoch time: 2 seconds as requested (e.g., -1.0 to 1.0)
        tmin, tmax = -1.0, 1.0
        
        epochs = mne.Epochs(raw, new_events, event_id=event_dict, tmin=tmin, tmax=tmax, 
                            baseline=None, preload=True, reject=None, verbose=False)
        
        # Drop EOG channels if they are not needed for connectivity
        # Usually PLV is between EEG channels.
        # Based on channels.tsv: HEOG_left, HEOG_right, VEOG_lower
        eog_chs = [ch for ch in raw.ch_names if 'EOG' in ch]
        if eog_chs:
            epochs.drop_channels(eog_chs)
            
        # Save epochs
        epochs_file = output_path / f"{subject_id}-epo.fif"
        epochs.save(str(epochs_file), overwrite=True, verbose=False)
        logger.info(f"Saved {epochs_file} - Epochs: {len(epochs)}")
        
    except Exception as e:
        logger.error(f"Error processing {subject_id}: {str(e)}")

if __name__ == "__main__":
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    output_path = base_path / "data" / "processed"
    output_path.mkdir(parents=True, exist_ok=True)
    
    subjects = [d for d in os.listdir(base_path) if d.startswith('sub-')]
    subjects.sort()
    
    # Process subjects
    for sub in subjects:
        preprocess_subject(sub, base_path, output_path)
