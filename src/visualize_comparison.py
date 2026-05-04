import mne
import matplotlib.pyplot as plt
from config import *

def visualize_comparison(sub_id='sub-001'):
    print(f"Bắt đầu trực quan hóa so sánh Before/After cho {sub_id}...")
    
    # 1. Load Raw Data (Tín hiệu gốc)
    raw_path = BASE_PATH / sub_id / "eeg" / f"{sub_id}_task-ERN_eeg.set"
    raw = mne.io.read_raw_eeglab(raw_path, preload=True, verbose=False)
    
    # Chuẩn hóa tên kênh để plot
    rename_map = {'FP1': 'Fp1', 'FP2': 'Fp2'}
    raw.rename_channels(lambda x: rename_map.get(x, x))
    raw.set_montage(MONTAGE_NAME, on_missing='ignore')
    
    # Chỉ lọc sơ bộ để dễ nhìn (0.1-30Hz) nhưng CHƯA chạy ICA/CSD
    raw_before = raw.copy().filter(0.1, 30.0, verbose=False)
    
    # 2. Load Processed Epochs (Tín hiệu sau khi chạy ICA & CSD)
    processed_path = EPOCHS_DIR / f"{sub_id}_master-epo.fif"
    epochs_after = mne.read_epochs(processed_path, preload=True, verbose=False)
    
    # Lấy trung bình (Evoked) để so sánh SNR
    evoked_after = epochs_after['Incorrect'].average()
    
    # 3. Visualization: Time Domain (So sánh đoạn tín hiệu)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)
    
    # Plot Before (Raw) - Một đoạn 2 giây đầu tiên
    start, stop = raw_before.time_as_index([10, 13]) # Lấy đoạn từ giây thứ 10 đến 13
    data_before, times_before = raw_before[:, start:stop]
    axes[0].plot(times_before, data_before[raw_before.ch_names.index('Fp1')].T * 1e6, label='Fp1 (Eye Channel)', alpha=0.7)
    axes[0].plot(times_before, data_before[raw_before.ch_names.index('FCz')].T * 1e6, label='FCz (Brain Channel)', alpha=0.9)
    axes[0].set_title(f"Trước khi xử lý (Raw + Filter): Thấy rõ nhiễu mắt (Eye Blinks) lớn ở Fp1 và FCz")
    axes[0].set_ylabel("Amplitude (μV)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot After (Cleaned Epochs) - Một trial ngẫu nhiên
    trial_idx = 0
    data_after = epochs_after.get_data(copy=True)[trial_idx] * 1e6
    times_after = epochs_after.times
    axes[1].plot(times_after, data_after[epochs_after.ch_names.index('Fp1')].T, label='Fp1 (Cleaned)', alpha=0.7)
    axes[1].plot(times_after, data_after[epochs_after.ch_names.index('FCz')].T, label='FCz (Cleaned)', alpha=0.9)
    axes[1].set_title(f"Sau khi xử lý (ICA + CSD): Nhiễu mắt đã bị loại bỏ, tín hiệu não FCz rõ nét hơn")
    axes[1].set_ylabel("Amplitude (μV)")
    axes[1].set_xlabel("Time (s) relative to response")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_compare = OUTPUTS_DIR / "comparison_time_domain.png"
    plt.savefig(output_compare)
    print(f"  Đã lưu so sánh Time Domain: {output_compare}")

    # 4. Visualization: Power Spectral Density (PSD)
    fig_psd, ax_psd = plt.subplots(figsize=(10, 6))
    raw_before.compute_psd(fmax=40).plot(axes=ax_psd, show=False, picks='FCz', color='red', spatial_colors=False)
    epochs_after.compute_psd(fmax=40).plot(axes=ax_psd, show=False, picks='FCz', color='blue', spatial_colors=False)
    
    ax_psd.set_title("So sánh Phổ năng lượng (PSD) tại FCz: Trước (Đỏ) vs Sau (Xanh)")
    ax_psd.legend(['Before (Raw)', 'After (Processed)'])
    
    output_psd = OUTPUTS_DIR / "comparison_psd.png"
    plt.savefig(output_psd)
    print(f"  Đã lưu so sánh PSD: {output_psd}")

if __name__ == "__main__":
    visualize_comparison()
