import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mne.preprocessing import compute_current_source_density
import os
from config import *

def visualize_preprocessing_comparison(raw_orig, raw_csd, sub_id):
    """Vẽ so sánh tín hiệu trước và sau khi tiền xử lý (No ICA, Yes CSD)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Butterfly Plot - Trước (Raw)
    data_orig = raw_orig.get_data() * 1e6
    times = raw_orig.times
    # Lấy 5 giây
    t_end = min(25, times[-1])
    t_start = max(0, t_end - 5)
    start, stop = raw_orig.time_as_index([t_start, t_end])
    
    axes[0, 0].plot(times[start:stop], data_orig[:, start:stop].T, color='black', alpha=0.3)
    axes[0, 0].set_title(f"{sub_id} - RAW (Butterfly)")
    axes[0, 0].set_ylabel("Amplitude (μV)")
    
    # 2. Butterfly Plot - Sau (CSD + Filter)
    data_csd = raw_csd.get_data() * 1e6
    axes[0, 1].plot(times[start:stop], data_csd[:, start:stop].T, color='teal', alpha=0.3)
    axes[0, 1].set_title(f"{sub_id} - PREPROCESSED (CSD + Filter)")
    axes[0, 1].set_ylabel("Amplitude (μV/mm²)")

    # 3. PSD - Trước
    raw_orig.compute_psd(fmax=45).plot(axes=axes[1, 0], show=False)
    axes[1, 0].set_title("PSD - RAW")

    # 4. PSD - Sau
    raw_csd.compute_psd(fmax=45).plot(axes=axes[1, 1], show=False)
    axes[1, 1].set_title("PSD - PREPROCESSED")

    plt.tight_layout()
    comp_path = OUTPUTS_DIR / f"preprocess_comp_{sub_id}.png"
    plt.savefig(comp_path)
    plt.close()
    print(f"  -> Đã lưu so sánh tiền xử lý: {comp_path}")

def preprocess_subject_ozdemir_style(sub_id, visualize=False):
    """
    Tiền xử lý theo Ozdemir (2017):
    - KHÔNG DÙNG ICA.
    - Dùng CSD (Current Source Density).
    """
    raw_path = DATA_RAW / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    if not raw_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {raw_path}")
        
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    
    # Rename & Montage (Làm trước khi copy để đồng bộ)
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw.rename_channels(lambda x: rename_map.get(x, x))
    raw.set_montage(MONTAGE_NAME, on_missing='ignore')
    
    # 2. Resampling & Filtering
    raw.resample(SFREQ)
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design='firwin', verbose=False)
    
    # 3. Pick Channels
    available_eeg = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
    raw_eeg_only = raw.copy().pick_channels(available_eeg)
    
    # 4. CSD
    raw_csd = compute_current_source_density(raw_eeg_only, verbose=False)
    
    if visualize:
        visualize_preprocessing_comparison(raw_eeg_only, raw_csd, sub_id)
    
    # 5. Epoching
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
    sub_ids = sorted([d.name for d in DATA_RAW.glob("sub-*") if d.is_dir()])
    
    report_data = []
    print(f"🚀 Chạy lại tiền xử lý (Ozdemir Style - NO ICA) cho {len(sub_ids)} subjects...")
    
    visualized = False
    for i, sub in enumerate(sub_ids):
        try:
            # Visualize cho subject đầu tiên thành công
            ep = preprocess_subject_ozdemir_style(sub, visualize=(not visualized))
            visualized = True # Đánh dấu đã visualize thành công một mẫu
            
            report_data.append({
                'sub': sub, 
                'status': 'SUCCESS', 
                'correct': len(ep['Correct']), 
                'incorrect': len(ep['Incorrect'])
            })
            if (i + 1) % 10 == 0:
                print(f"  Done: {i+1}/{len(sub_ids)} subjects...")
                
        except Exception as e:
            report_data.append({'sub': sub, 'status': f'FAILED: {e}', 'correct': 0, 'incorrect': 0})
            print(f"  ❌ FAILED: {sub} | {e}")
            
    pd.DataFrame(report_data).to_csv(REPORT_ETL, index=False)
    print(f"\n✅ HOÀN TẤT! Báo cáo: {REPORT_ETL}")
