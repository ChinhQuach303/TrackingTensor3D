import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import *

def visualize_grand_average():
    print("Bắt đầu trực quan hóa kết quả tiền xử lý...")
    
    # 1. Tìm các tệp đã xử lý
    sub_files = list(EPOCHS_DIR.glob("*_master-epo.fif"))
    if not sub_files:
        print("Lỗi: Không tìm thấy tệp epoch nào. Vui lòng chạy preprocessing trước.")
        return

    all_evokeds_inc = []
    all_evokeds_cor = []

    print(f"  Đang tổng hợp dữ liệu từ {len(sub_files)} đối tượng...")
    for f in sub_files:
        epochs = mne.read_epochs(f, preload=True, verbose=False)
        all_evokeds_inc.append(epochs['Incorrect'].average())
        all_evokeds_cor.append(epochs['Correct'].average())

    # Grand Average
    ga_inc = mne.grand_average(all_evokeds_inc)
    ga_cor = mne.grand_average(all_evokeds_cor)

    # 2. Plot Grand Average ERP tại FCz (Kênh trọng tâm của ERN)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # MNE plot_compare_evokeds
    mne.viz.plot_compare_evokeds(
        {'Incorrect (ERN)': ga_inc, 'Correct (CRN)': ga_cor},
        picks='FCz',
        axes=ax,
        show=False,
        title='Grand Average ERP tại kênh FCz (Sau khi xử lý ICA & CSD)',
        colors={'Incorrect (ERN)': 'red', 'Correct (CRN)': 'blue'},
        linestyles={'Incorrect (ERN)': '-', 'Correct (CRN)': '--'}
    )
    
    # Nghịch đảo trục Y (truyền thống EEG: âm lên trên)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    
    output_erp = OUTPUTS_DIR / "grand_average_erp_fcz.png"
    plt.savefig(output_erp)
    print(f"  Đã lưu biểu đồ ERP GA: {output_erp}")

    # 3. Plot Topomap tại đỉnh ERN (khoảng 50ms)
    fig_topo, axes_topo = plt.subplots(1, 2, figsize=(12, 5))
    
    # Tìm thời điểm đỉnh ERN (thường là âm nhất trong khoảng 0-100ms)
    times = ga_inc.times
    mask = (times > 0.02) & (times < 0.1)
    peak_time = times[mask][np.argmin(ga_inc.copy().pick('FCz').data[0][mask])]
    
    ga_inc.plot_topomap(times=peak_time, axes=axes_topo[0], show=False, colorbar=False)
    axes_topo[0].set_title(f'Incorrect (ERN) Topomap\nat {peak_time*1000:.0f} ms')
    
    ga_cor.plot_topomap(times=peak_time, axes=axes_topo[1], show=False, colorbar=False)
    axes_topo[1].set_title(f'Correct (CRN) Topomap\nat {peak_time*1000:.0f} ms')
    
    output_topo = OUTPUTS_DIR / "grand_average_topomaps.png"
    plt.savefig(output_topo)
    print(f"  Đã lưu bản đồ Topo GA: {output_topo}")

if __name__ == "__main__":
    visualize_grand_average()
