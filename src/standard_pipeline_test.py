import mne
import numpy as np
import pandas as pd
from pathlib import Path
from mne.preprocessing import ICA

# Cấu hình
base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
sub_id = "sub-002"
set_file = base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"

# 1. Load và Lọc
raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
raw.filter(1, 40, fir_design='firwin', verbose=False)

# 2. Thiết lập Montage (Tọa độ) và Dò kênh lỗi
print("Step 2: Standardizing names and Setting Montage...")
# Chuẩn hóa tên kênh để khớp với thư viện MNE (ví dụ FP1 -> Fp1)
mapping = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 
    'HEOG_left': 'EOG1', 'HEOG_right': 'EOG2', 'VEOG_lower': 'EOG3'
}
raw.rename_channels(lambda x: mapping.get(x, x))

montage = mne.channels.make_standard_montage('standard_1020')
# Gán montage
raw.set_montage(montage, on_missing='ignore')
# LƯU BẢN ĐẦY ĐỦ Ở ĐÂY (trước khi lọc bớt kênh)
raw_full = raw.copy()

# Loại bỏ những kênh EEG nào không gán được tọa độ (để bảo vệ bước nội suy)
chs_with_pos = [ch for ch in raw.ch_names if not np.isnan(raw.get_montage().get_positions()['ch_pos'].get(ch, [np.nan])[0])]
raw.pick(chs_with_pos)

# Dò lỗi
eeg_data = raw.get_data(picks='eeg')
stds = np.std(eeg_data, axis=1)
med_std = np.median(stds)
bad_idx = np.where((stds < 1e-7) | (stds > 5 * med_std))[0]
eeg_names = raw.copy().pick('eeg').ch_names
bad_names = [eeg_names[i] for i in bad_idx]
print(f"   Detected Bad Channels: {bad_names}")
raw.info['bads'] = bad_names

# 3. Nội suy kênh hỏng
if bad_names:
    print(f"   Interpolating bad channels: {bad_names}")
    raw.interpolate_bads(reset_bads=True)

# 4. Tham chiếu lại (Re-referencing to average)
print("Step 4: Re-referencing to Average...")
# raw lúc này chỉ còn EEG, ta có thể đặt tham chiếu trung bình ngay
raw.set_eeg_reference('average', projection=False, verbose=False)

# 5. Chạy ICA
print("Step 5: Running ICA...")
ica = ICA(n_components=20, random_state=42, method='infomax', fit_params=dict(extended=True), verbose=False)
ica.fit(raw, verbose=False)
# Tự động tìm nhiễu mắt bằng cách đối chiếu với kênh EOG3 (đã đổi tên)
bad_ids, scores = ica.find_bads_eog(raw_full, ch_name='EOG3', verbose=False)
ica.exclude = bad_ids
print(f"   ICA identified {len(bad_ids)} eye artifact components: {bad_ids}")
raw_cleaned = raw.copy()
ica.apply(raw_cleaned)

# 6. Kiểm tra lại kết quả
# Load events
events_df = pd.read_csv(base_path / sub_id / "eeg" / f"{sub_id}_task-ERN_events.tsv", sep='\t')
event_codes = [111, 212, 121, 222, 112, 221, 122, 211]
mne_events = np.array([[int(round(row['onset'] * raw.info['sfreq'])), 0, int(row['value'])] 
                       for _, row in events_df.iterrows() if int(row['value']) in event_codes])

epochs_clean = mne.Epochs(raw_cleaned, mne_events, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0), preload=True, verbose=False)

# Tính tỉ lệ Bad Trials mới
ptp_clean = (np.max(epochs_clean.get_data(picks='eeg'), axis=-1) - np.min(epochs_clean.get_data(picks='eeg'), axis=-1)) * 1e6
bad_rate = np.mean(np.any(ptp_clean > 100, axis=1)) * 100

print(f"\nFINISH: Final Artifact Rate for {sub_id}: {bad_rate:.2f}%")
