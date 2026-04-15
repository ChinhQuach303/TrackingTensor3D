import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
epochs_dir = base_path / "data" / "processed" / "epochs"

# Danh sách các subject đã hoàn thành (tính đến sub-015)
cleaned_files = sorted([f for f in os.listdir(epochs_dir) if f.endswith("_clean-epo.fif")])

all_evoked_correct = []
all_evoked_incorrect = []

print(f"Aggregating {len(cleaned_files)} cleaned subjects...")

for f in cleaned_files:
    try:
        epochs = mne.read_epochs(epochs_dir / f, preload=True, verbose=False)
        all_evoked_correct.append(epochs['Correct'].average())
        all_evoked_incorrect.append(epochs['Incorrect'].average())
    except:
        continue

if all_evoked_correct:
    ga_correct = mne.grand_average(all_evoked_correct)
    ga_incorrect = mne.grand_average(all_evoked_incorrect)
    
    ch_idx = ga_correct.ch_names.index('FCz')
    times = ga_correct.times * 1000
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, ga_correct.data[ch_idx]*1e6, label=f'Grand Average Correct (N={len(all_evoked_correct)})', color='blue')
    plt.plot(times, ga_incorrect.data[ch_idx]*1e6, label=f'Grand Average Incorrect (N={len(all_evoked_incorrect)})', color='red', linewidth=2)
    plt.axvline(0, color='black', linestyle='-')
    plt.gca().invert_yaxis()
    plt.title(f'Grand Average ERP at FCz (Cleaned Subjects up to sub-015)\nBaseline: -600ms to -400ms')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (uV)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(base_path / "outputs" / "eda" / "grand_average_v15.png")
    print(f"Grand Average plot saved to outputs/eda/grand_average_v15.png")
else:
    print("No evoked data found.")
