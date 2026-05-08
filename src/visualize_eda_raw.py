import numpy as np
import mne
import matplotlib.pyplot as plt
from config import *
from pathlib import Path

def visualize_single_sub_eda(raw, sub_id):
    """Vẽ biểu đồ Butterfly và Offset cho một đối tượng mẫu."""
    # Lọc nhẹ để quan sát sóng
    raw_filt = raw.copy().filter(0.1, 30.0, verbose=False)
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 12))
    
    # 1. Butterfly plot
    data = raw_filt.get_data() * 1e6 # Chuyển sang microVolt
    times = raw_filt.times
    # Lấy 10 giây đoạn giữa (giả sử file dài hơn 30s)
    t_start, t_stop = 20, 30
    if times[-1] < 30: t_start, t_stop = 0, min(10, times[-1])
    
    start, stop = raw_filt.time_as_index([t_start, t_stop])
    
    ax[0].plot(times[start:stop], data[:, start:stop].T, color='black', alpha=0.3, linewidth=0.5)
    ax[0].set_title(f"Butterfly Plot ({sub_id}): Quan sát nhiễu đồng bộ trên 30 kênh")
    ax[0].set_ylabel("Amplitude (μV)")
    ax[0].grid(True, alpha=0.2)
    
    # 2. Offset plot cho các kênh đại diện
    picks = ['Fp1', 'Fz', 'F3', 'F4', 'FCz', 'Cz', 'Pz', 'Oz', 'HEOG_left']
    picks = [ch for ch in picks if ch in raw_filt.ch_names]
    offset = 60 
    
    for i, ch in enumerate(picks):
        ch_idx = raw_filt.ch_names.index(ch)
        ax[1].plot(times[start:stop], data[ch_idx, start:stop] + (i * offset), label=ch)
    
    ax[1].set_title(f"Tín hiệu đại diện (Offset): Kiểm tra nhiễu mắt (Fp1) và nhiễu kênh")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_yticks(np.arange(len(picks)) * offset)
    ax[1].set_yticklabels(picks)
    ax[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    output_path = OUTPUTS_DIR / f"eda_raw_{sub_id}.png"
    plt.savefig(output_path)
    print(f"  -> Đã lưu EDA cá thể: {output_path}")

def run_global_eda():
    """Chạy EDA cho toàn bộ 40 đối tượng và vẽ Grand Average PSD."""
    sub_ids = sorted([d.name for d in DATA_RAW.glob("sub-*") if d.is_dir()])
    print(f"🚀 Bắt đầu EDA tổng quát cho {len(sub_ids)} subjects...")
    
    all_psds = []
    freqs = None
    
    for i, sub_id in enumerate(sub_ids):
        try:
            raw_path = DATA_RAW / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
            if not raw_path.exists():
                continue
                
            raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
            
            # Standardize channels
            rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
            raw.rename_channels(lambda x: rename_map.get(x, x))
            raw.set_montage(MONTAGE_NAME, on_missing='ignore')
            
            # Chỉ lấy các kênh EEG chính để tính PSD
            eeg_picks = [ch for ch in EEG_CHANNELS if ch in raw.ch_names]
            raw_eeg = raw.copy().pick_channels(eeg_picks)
            
            # Tính PSD (Power Spectral Density)
            psd_obj = raw_eeg.compute_psd(fmin=1, fmax=45, verbose=False)
            psd_data, freqs_obj = psd_obj.get_data(return_freqs=True)
            
            # Trung bình qua các kênh
            avg_psd = np.mean(psd_data, axis=0)
            all_psds.append(avg_psd)
            freqs = freqs_obj
            
            # Vẽ minh họa chi tiết cho đối tượng đầu tiên
            if i == 0:
                visualize_single_sub_eda(raw, sub_id)
                
            if (i + 1) % 10 == 0:
                print(f"  Đã xử lý {i+1}/{len(sub_ids)} subjects...")
                
        except Exception as e:
            print(f"  ⚠️ Lỗi tại {sub_id}: {e}")

    # 3. Vẽ Grand Average PSD
    if all_psds:
        all_psds = np.array(all_psds)
        mean_psd = np.mean(all_psds, axis=0)
        std_psd = np.std(all_psds, axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, 10 * np.log10(mean_psd), color='teal', linewidth=2, label='Grand Average Mean')
        plt.fill_between(freqs, 
                         10 * np.log10(mean_psd - std_psd/2), 
                         10 * np.log10(mean_psd + std_psd/2), 
                         color='teal', alpha=0.2, label='Inter-subject Variation')
        
        plt.title(f"Grand Average PSD (N={len(all_psds)}) - Toàn bộ 40 Subjects")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB/Hz)")
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        
        output_grand = OUTPUTS_DIR / "grand_average_psd_raw.png"
        plt.savefig(output_grand)
        print(f"\n✅ HOÀN TẤT EDA TỔNG QUÁT!")
        print(f"📊 Grand Average PSD lưu tại: {output_grand}")
    else:
        print("❌ Không tìm thấy dữ liệu để tính Grand Average.")

if __name__ == "__main__":
    run_global_eda()
