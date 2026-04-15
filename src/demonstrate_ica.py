import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA

# Cấu hình
base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
sub_id = "sub-001"
set_file = base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
out_path = base_path / "outputs" / "eda"
out_path.mkdir(parents=True, exist_ok=True)

# 1. Load data
raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
raw.filter(1, 40, fir_design='firwin', verbose=False)

# Xác định kênh EOG và EEG
eog_chs = [ch for ch in raw.ch_names if 'EOG' in ch]
eeg_chs = [ch for ch in raw.ch_names if ch not in eog_chs]

# 2. Run ICA
# Chúng ta chỉ chạy ICA trên các kênh EEG để tìm nhiễu mắt lọt vào đó
ica = ICA(n_components=20, random_state=42, method='infomax', fit_params=dict(extended=True), verbose=False)
ica.fit(raw, picks='eeg')

# 3. Tìm thành phần nhiễu mắt (theo tương quan với kênh VEOG/HEOG)
# Dùng VEOG_lower để tìm chớp mắt
bad_ids, scores = ica.find_bads_eog(raw, ch_name='VEOG_lower', verbose=False)
ica.exclude = bad_ids

# 4. Tạo bản sao dữ liệu đã lọc sạch
raw_cleaned = raw.copy()
ica.apply(raw_cleaned)

# 5. Visualize kết quả (Tập trung vào ERP)
# Xử lý events để so sánh
events_df = pd.read_csv(base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_events.tsv", sep='\t')
mne_events = []
for _, row in events_df.iterrows():
    val = int(row['value'])
    if val in [111, 212, 121, 222]: new_val = 1 # Correct
    elif val in [112, 221, 122, 211]: new_val = 2 # Incorrect
    else: continue
    mne_events.append([int(round(row['onset'] * raw.info['sfreq'])), 0, new_val])
mne_events = np.array(mne_events)

# Epoching cả 2 bản (Thô vs Sạch)
event_id = {'Correct': 1, 'Incorrect': 2}
epochs_raw = mne.Epochs(raw, mne_events, event_id=event_id, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, verbose=False)
epochs_clean = mne.Epochs(raw_cleaned, mne_events, event_id=event_id, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, verbose=False)

# Plot so sánh Erroneous trials (Incorrect) tại FCz
evoked_raw = epochs_raw['Incorrect'].average()
evoked_clean = epochs_clean['Incorrect'].average()
times = evoked_raw.times * 1000
ch_idx = evoked_raw.ch_names.index('FCz')

plt.figure(figsize=(10, 6))
plt.plot(times, evoked_raw.data[ch_idx]*1e6, label='Before ICA (Raw)', color='gray', alpha=0.5, linestyle='--')
plt.plot(times, evoked_clean.data[ch_idx]*1e6, label='After ICA (Cleaned ERN)', color='red', linewidth=2)
plt.axvline(0, color='black', linestyle='-')
plt.gca().invert_yaxis()
plt.title(f'Effect of ICA Cleaning on ERN signal (Sub-001, Channel FCz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (uV)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(out_path / "ica_comparison_sub001.png")

# Kiểm tra lại tỉ lệ nhiễu sau lọc
data_clean = epochs_clean.get_data(picks='eeg') * 1e6
ptp_clean = np.max(data_clean, axis=-1) - np.min(data_clean, axis=-1)
new_bad_rate = np.mean(np.any(ptp_clean > 100, axis=1)) * 100
print(f'New Artifact Rate after ICA: {new_bad_rate:.2f}%')
