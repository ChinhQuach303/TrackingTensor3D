import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from config import *

def run_statistical_validation():
    print("Bắt đầu Kiểm định Thống kê (Statistical Validation)...")
    
    # 1. Load Data
    w_inc = np.load(TENSOR_DIR / "horls_weights_incorrect.npy")
    w_cor = np.load(TENSOR_DIR / "horls_weights_correct.npy")
    energy_inc = np.load(TENSOR_DIR / "horls_energy_incorrect.npy") # This was collective energy
    
    # We need subject-level energy for t-tests
    # Let's load the 4D tensors to get individual network energy evolution
    t_inc = np.load(TENSOR_CORRECT_FILE.parent / "tensor_incorrect_4d.npy") # (34, 30, 30, 256)
    t_cor = np.load(TENSOR_CORRECT_FILE.parent / "tensor_correct_4d.npy")
    
    n_subs = t_inc.shape[0]
    sub_energy_inc = np.zeros((n_subs, 256))
    sub_energy_cor = np.zeros((n_subs, 256))
    
    print("  Calculating subject-level network energy...")
    for s in range(n_subs):
        for t in range(256):
            # Energy = norm of (C_t * w_common_t)
            sub_energy_inc[s, t] = np.linalg.norm(t_inc[s, :, :, t] @ w_inc[:, t])
            sub_energy_cor[s, t] = np.linalg.norm(t_cor[s, :, :, t] @ w_cor[:, t])
            
    # 2. Point-wise T-Test (Correct vs Incorrect)
    t_stats, p_values = stats.ttest_rel(sub_energy_inc, sub_energy_cor, axis=0)
    
    # 3. Behavioral Correlation
    report = pd.read_csv(REPORT_ETL)
    # Filter only success subjects who are in our tensor (n_incorrect >= 15)
    valid_report = report[report['incorrect'] >= MIN_INCORRECT_TRIALS].copy()
    valid_report['error_rate'] = valid_report['incorrect'] / (valid_report['correct'] + valid_report['incorrect'])
    
    # Peak ERN energy (25-100ms) per subject
    peak_ern_energy = np.max(sub_energy_inc[:, 128+3 : 128+13], axis=1)
    r_val, p_val_corr = stats.pearsonr(peak_ern_energy, valid_report['error_rate'])
    
    # 4. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # A. Energy Differences with Significance
    time_ms = np.linspace(-1000, 1000, 256)
    axes[0].plot(time_ms, np.mean(sub_energy_inc, axis=0), color='red', label='Incorrect Energy')
    axes[0].plot(time_ms, np.mean(sub_energy_cor, axis=0), color='gray', linestyle='--', label='Correct Energy')
    
    # Highlight significant regions (p < 0.05)
    sig_mask = p_values < 0.05
    axes[0].fill_between(time_ms, 0, np.max(sub_energy_inc), where=sig_mask, color='yellow', alpha=0.2, label='p < 0.05')
    
    axes[0].set_title("Statistical Comparison of Network Energy (Incorrect vs Correct)")
    axes[0].set_ylabel("Collective Network Energy")
    axes[0].axvline(0, color='black')
    axes[0].legend()
    
    # B. Correlation Plot
    axes[1].scatter(valid_report['error_rate'], peak_ern_energy, color='red', alpha=0.6)
    # Regression line
    m, b = np.polyfit(valid_report['error_rate'], peak_ern_energy, 1)
    axes[1].plot(valid_report['error_rate'], m*valid_report['error_rate'] + b, color='black', label=f'R={r_val:.3f}, p={p_val_corr:.3f}')
    
    axes[1].set_title("Behavioral Correlation: ERN Network Strength vs Error Rate")
    axes[1].set_xlabel("Error Rate")
    axes[1].set_ylabel("Peak ERN Network Energy (4-8Hz)")
    axes[1].legend()
    
    plt.tight_layout()
    output_img = OUTPUTS_DIR / "statistical_validation_final.png"
    plt.savefig(output_img)
    print(f"Statistical validation complete. Results saved to: {output_img}")
    print(f"Peak Correlation R: {r_val:.3f} (p={p_val_corr:.3f})")

if __name__ == "__main__":
    run_statistical_validation()
