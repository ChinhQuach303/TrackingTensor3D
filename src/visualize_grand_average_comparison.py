import mne
import numpy as np
import matplotlib.pyplot as plt
from config import *

def get_raw_evoked(sub_id):
    # Hàm tạo Evoked từ Raw (Chỉ lọc, không ICA/CSD)
    raw_path = BASE_PATH / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw.rename_channels(lambda x: rename_map.get(x, x))
    raw.set_montage(MONTAGE_NAME, on_missing='ignore')
    raw.resample(SFREQ)
    raw.filter(0.1, 30.0, verbose=False)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    incorrect_codes = [l for l in event_id.keys() if len(l) == 3 and l[0] != l[2]]
    resp_id = {label: event_id[label] for label in incorrect_codes}
    
    epochs = mne.Epochs(raw, events, event_id=resp_id, tmin=TMIN, tmax=TMAX, 
                        baseline=BASELINE, preload=True, verbose=False)
    return epochs.average()

def visualize_grand_comparison():
    print("Bắt đầu so sánh Grand Average (Raw vs Cleaned)...")
    sub_ids = [f.name.split('_')[0] for f in list(EPOCHS_DIR.glob("*.fif"))[:10]]
    
    all_raw_inc = []
    all_clean_inc = []
    
    for sid in sub_ids:
        print(f"  Đang lấy dữ liệu {sid}...")
        try:
            # 1. Raw Evoked (Chỉ lọc)
            all_raw_inc.append(get_raw_evoked(sid))
            
            # 2. Cleaned Evoked (ICA + CSD)
            cleaned_path = EPOCHS_DIR / f"{sid}_master-epo.fif"
            all_clean_inc.append(mne.read_epochs(cleaned_path, verbose=False)['Incorrect'].average())
        except Exception as e:
            print(f"  Bỏ qua {sid}: {e}")

    # Ensure common channels for comparison
    common_chs = list(set(all_raw_inc[0].ch_names) & set(all_clean_inc[0].ch_names))
    all_raw_inc = [e.pick_channels(common_chs) for e in all_raw_inc]
    all_clean_inc = [e.pick_channels(common_chs) for e in all_clean_inc]

    ga_raw = mne.grand_average(all_raw_inc)
    ga_clean = mne.grand_average(all_clean_inc)
    
    # --- PLOT 1: ERP Waveforms tại 3 vị trí (Fz, FCz, Pz) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    locations = ['Fz', 'FCz', 'Pz']
    
    for i, loc in enumerate(locations):
        mne.viz.plot_compare_evokeds(
            {'Raw (Blink Contaminated)': ga_raw, 'Cleaned (ICA + CSD)': ga_clean},
            picks=loc, axes=axes[i], show=False,
            title=f'Grand Average ERP tại {loc}'
        )
        axes[i].invert_yaxis()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "grand_average_comparison_waveforms.png")
    
    # --- PLOT 2: Topomap Comparison tại đỉnh ERN (~60ms) ---
    fig_topo, axes_topo = plt.subplots(2, 1, figsize=(8, 10))
    peak_time = 0.06 # 60ms
    
    # Raw Topo
    ga_raw.plot_topomap(times=peak_time, axes=axes_topo[0], show=False, colorbar=False)
    axes_topo[0].set_title(f"Raw Grand Average Topomap (60ms)\nBị nhiễu mắt ở vùng trán làm át tín hiệu não")
    
    # Cleaned Topo
    ga_clean.plot_topomap(times=peak_time, axes=axes_topo[1], show=False, colorbar=False)
    axes_topo[1].set_title(f"Cleaned Grand Average Topomap (60ms)\nTín hiệu ERN tập trung rõ rệt tại vùng trung tâm-trán")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "grand_average_comparison_topomaps.png")
    print(f"Hoàn tất! Kết quả lưu tại {OUTPUTS_DIR}")

if __name__ == "__main__":
    visualize_grand_comparison()
