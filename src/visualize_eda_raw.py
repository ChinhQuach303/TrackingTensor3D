import numpy as np
import mne
import matplotlib.pyplot as plt
from config import *

def visualize_raw_eda(sub_id='sub-001'):
    print(f"Bắt đầu EDA tín hiệu thô đa kênh cho {sub_id}...")
    
    # 1. Load Raw Data
    raw_path = BASE_PATH / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    
    # Rename & Montage
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw.rename_channels(lambda x: rename_map.get(x, x))
    raw.set_montage(MONTAGE_NAME, on_missing='ignore')
    
    # Lọc 0.1-30Hz để có thể nhìn thấy sóng não (không thì drift quá lớn khó nhìn)
    # Nhưng giữ nguyên nhiễu mắt và nhiễu kênh
    raw_filt = raw.copy().filter(0.1, 30.0, verbose=False)
    
    # 2. Plot Butterfly (Tất cả các kênh chồng lên nhau)
    fig, ax = plt.subplots(2, 1, figsize=(15, 12))
    
    # Butterfly plot
    data = raw_filt.get_data() * 1e6
    times = raw_filt.times
    start, stop = raw_filt.time_as_index([20, 30]) # Lấy 10 giây đoạn giữa
    
    ax[0].plot(times[start:stop], data[:, start:stop].T, color='black', alpha=0.3, linewidth=0.5)
    ax[0].set_title(f"Butterfly Plot (Tất cả 30 kênh): Thấy rõ các đỉnh nhiễu đồng bộ (thường là mắt/chuyển động)")
    ax[0].set_ylabel("Amplitude (μV)")
    ax[0].grid(True, alpha=0.2)
    
    # 3. Plot 15 kênh đại diện (Phân tầng)
    picks = ['Fp1', 'Fz', 'F3', 'F4', 'FCz', 'Cz', 'C3', 'C4', 'Pz', 'P3', 'P4', 'Oz', 'O1', 'O2', 'HEOG_left']
    picks = [ch for ch in picks if ch in raw_filt.ch_names]
    
    # Offset plotting
    offset = 50 # 50 uV offset
    for i, ch in enumerate(picks):
        ch_idx = raw_filt.ch_names.index(ch)
        ax[1].plot(times[start:stop], data[ch_idx, start:stop] + (i * offset), label=ch)
    
    ax[1].set_title(f"Tín hiệu 15 kênh đại diện (Offset): Nhiễu mắt cực mạnh tại các kênh Trán (Fp1, Fz)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_yticks(np.arange(len(picks)) * offset)
    ax[1].set_yticklabels(picks)
    ax[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    output_eda = OUTPUTS_DIR / "eda_raw_multichannel.png"
    plt.savefig(output_eda)
    print(f"  Đã lưu EDA đa kênh: {output_eda}")

if __name__ == "__main__":
    visualize_raw_eda()
