import mne
import numpy as np
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA

def process_pilot(sub_id, base_path):
    print(f"\n--- Processing Pilot: {sub_id} ---")
    set_file = base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    
    # 1. Load data
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    
    # 2. Standardize Names & Montage
    mapping = {'FP1': 'Fp1', 'FP2': 'Fp2', 'VEOG_lower': 'EOG3'} 
    raw.rename_channels(lambda x: mapping.get(x, x))
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    # 3. Detect Bad Channels (Mark only)
    print("   Step 3: Detecting Bad Channels...")
    eeg_data = raw.get_data(picks='eeg')
    stds = np.std(eeg_data, axis=1)
    bad_idx = np.where((stds < 1e-7) | (stds > 5 * np.median(stds)))[0]
    eeg_names = raw.copy().pick('eeg').ch_names
    bad_names = [eeg_names[i] for i in bad_idx]
    raw.info['bads'] = bad_names
    print(f"      Bads marked: {bad_names}")

    # 4. Filter 1-40 Hz
    raw.filter(1, 40, fir_design='firwin', verbose=False)

    # 5. Average Reference
    print("   Step 5: Average Referencing...")
    raw_full = raw.copy() 
    raw.pick('eeg') 
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # 6. ICA
    print("   Step 6: Running ICA...")
    ica = ICA(n_components=15, random_state=42, method='infomax', fit_params=dict(extended=True), verbose=False)
    ica.fit(raw, verbose=False)
    
    # Find Bads EOG
    bad_ids, scores = ica.find_bads_eog(raw_full, ch_name='EOG3', verbose=False)
    ica.exclude = bad_ids
    print(f"      ICA excluded {len(bad_ids)} components: {bad_ids}")
    ica.apply(raw)

    # 7. Post-ICA Interpolation
    print("   Step 7: Interpolating Bad Channels...")
    raw.interpolate_bads(reset_bads=True, verbose=False)

    # 8. Epoching with the new Baseline (-600 to -400ms)
    print("   Step 8: Epoching with Baseline (-0.6, -0.4)...")
    events_df = pd.read_csv(base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_events.tsv", sep='\t')
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
                        tmin=-0.6, tmax=0.8, baseline=(-0.6, -0.4), preload=True, verbose=False)
    
    return epochs

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    pilots = ["sub-001", "sub-002", "sub-003"]
    all_epochs = []
    
    for sub in pilots:
        try:
            ep = process_pilot(sub, base_path)
            all_epochs.append(ep)
        except Exception as e:
            print(f"Error in {sub}: {e}")

    # Compute Grand Average
    if all_epochs:
        print("\n--- Generating Grand Average Plots ---")
        ga_correct = mne.grand_average([e['Correct'].average() for e in all_epochs])
        ga_incorrect = mne.grand_average([e['Incorrect'].average() for e in all_epochs])
        
        ch_idx = ga_correct.ch_names.index('FCz')
        times = ga_correct.times * 1000
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, ga_correct.data[ch_idx]*1e6, label='Grand Average Correct (CRN)', color='blue')
        plt.plot(times, ga_incorrect.data[ch_idx]*1e6, label='Grand Average Incorrect (ERN)', color='red', linewidth=2)
        plt.axvline(0, color='black', linestyle='-')
        plt.gca().invert_yaxis()
        plt.title('Grand Average ERP at FCz (sub-001 to sub-003)\nBaseline: -600ms to -400ms')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude (uV)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible/outputs/eda/ga_pilot_erp.png")
        print("Plot saved to outputs/eda/ga_pilot_erp.png")
