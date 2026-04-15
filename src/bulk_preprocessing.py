import mne
import numpy as np
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA
import os

def clean_subject(sub_id, base_path, output_dir):
    set_file = base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    events_file = base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_events.tsv"
    
    # 1. Load data
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    
    # 2. Standardize Names & Montage
    mapping = {'FP1': 'Fp1', 'FP2': 'Fp2', 'VEOG_lower': 'EOG3'} 
    raw.rename_channels(lambda x: mapping.get(x, x))
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    # 3. Detect Bad Channels
    eeg_data = raw.get_data(picks='eeg')
    stds = np.std(eeg_data, axis=1)
    bad_idx = np.where((stds < 1e-7) | (stds > 5 * np.median(stds)))[0]
    eeg_names = raw.copy().pick('eeg').ch_names
    bad_names = [eeg_names[i] for i in bad_idx]
    raw.info['bads'] = bad_names

    # 4. Filter 1-40 Hz
    raw.filter(1, 40, fir_design='firwin', verbose=False)

    # 5. Average Reference
    raw_full = raw.copy() 
    raw.pick('eeg') 
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # 6. ICA
    ica = ICA(n_components=15, random_state=42, method='infomax', fit_params=dict(extended=True), verbose=False)
    ica.fit(raw, verbose=False)
    bad_ids, _ = ica.find_bads_eog(raw_full, ch_name='EOG3', verbose=False)
    ica.exclude = bad_ids
    ica.apply(raw)

    # 7. Post-ICA Interpolation
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # 8. Epoching -1.0 to 1.0 around Response
    events_df = pd.read_csv(events_file, sep='\t')
    event_codes = {'Correct': [111, 212, 121, 222], 'Incorrect': [112, 221, 122, 211]}
    mne_events = []
    for _, row in events_df.iterrows():
        val = int(row['value'])
        if val in event_codes['Correct']: code = 1
        elif val in event_codes['Incorrect']: code = 2
        else: continue
        mne_events.append([int(round(row['onset'] * raw.info['sfreq'])), 0, code])
    mne_events = np.array(mne_events)
    
    epochs = mne.Epochs(raw, mne_events, event_id={'Correct': 1, 'Incorrect': 2}, 
                        tmin=-1.0, tmax=1.0, baseline=(-0.6, -0.4), preload=True, verbose=False)
    
    # Save results
    save_file = output_dir / f"{sub_id}_clean-epo.fif"
    epochs.save(save_file, overwrite=True, verbose=False)
    return len(epochs), np.mean(np.any((np.max(epochs.get_data()*1e6, axis=-1) - np.min(epochs.get_data()*1e6, axis=-1)) > 100, axis=1)) * 100

if __name__ == "__main__":
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    output_dir = base_path / "data" / "processed" / "epochs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = sorted([d for d in os.listdir(base_path) if d.startswith("sub-")])
    stats = []
    
    for sub in subjects:
        try:
            num_epochs, bad_rate = clean_subject(sub, base_path, output_dir)
            print(f"DONE: {sub} | Epochs: {num_epochs} | Bad Rate: {bad_rate:.2f}%")
            stats.append({'sub': sub, 'status': 'OK', 'bad_rate': bad_rate})
        except Exception as e:
            print(f"FAILED: {sub} | Error: {str(e)}")
            stats.append({'sub': sub, 'status': 'FAILED', 'error': str(e)})
    
    df = pd.DataFrame(stats)
    df.to_csv(base_path / "data" / "processed" / "cleaning_report.csv", index=False)
    print("\nProcessing complete. Report saved to data/processed/cleaning_report.csv")
