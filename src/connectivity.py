import mne
import numpy as np
import scipy.signal as signal
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_plv(epochs):
    """
    Compute Phase Locking Value (PLV) across trials.
    PLV[i, j, t] = |mean_k(exp(j*(phi_i[k,t] - phi_j[k,t])))|
    """
    data = epochs.get_data() # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = data.shape
    
    # Compute analytic signal using Hilbert transform
    logger.info(f"  Performing Hilbert transform on {n_epochs} trials...")
    analytic = signal.hilbert(data, axis=-1)
    phase = np.angle(analytic)
    
    # Pre-calculate exp(j * phase) to optimize
    phase_exp = np.exp(1j * phase)
    
    plv = np.zeros((n_channels, n_channels, n_times), dtype=np.float32)
    
    # Vectorized computation for each channel pair
    # We'll result in a lower triangular matrix first or just fill it
    for i in range(n_channels):
        # We can optimize by computing one channel vs all others at once
        # exp_i is (n_epochs, n_times)
        exp_i = phase_exp[:, i, :]
        # exp_others is (n_epochs, n_channels, n_times)
        # diff_phase[k, j, t] = exp_i[k, t] * conj(exp_j[k, t])
        # mean over k -> Result (n_channels, n_times)
        diff_phase = exp_i[:, np.newaxis, :] * np.conj(phase_exp)
        plv_row = np.abs(np.mean(diff_phase, axis=0))
        plv[i, :, :] = plv_row
        
    return plv

def process_all_connectivity(processed_path, output_path):
    files = list(processed_path.glob("sub-*-epo.fif"))
    files.sort()
    
    if not files:
        logger.error("No processed epoch files found.")
        return

    for f in files:
        sub_id = f.stem.split('-epo')[0]
        logger.info(f"Processing connectivity for {sub_id}...")
        try:
            epochs = mne.read_epochs(f, preload=True, verbose=False)
            
            # Correct trials
            if 'Correct' in epochs.event_id:
                logger.info(f"  Correct trials: {len(epochs['Correct'])}")
                plv_correct = compute_plv(epochs['Correct'])
                np.save(output_path / f"{sub_id}_correct_plv.npy", plv_correct)
                
            # Incorrect trials
            if 'Incorrect' in epochs.event_id:
                logger.info(f"  Incorrect trials: {len(epochs['Incorrect'])}")
                plv_incorrect = compute_plv(epochs['Incorrect'])
                np.save(output_path / f"{sub_id}_incorrect_plv.npy", plv_incorrect)
        except Exception as e:
            logger.error(f"  Error processing {sub_id}: {str(e)}")

if __name__ == "__main__":
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    processed_path = base_path / "data" / "processed"
    output_path = base_path / "outputs" / "connectivity"
    output_path.mkdir(parents=True, exist_ok=True)
    
    process_all_connectivity(processed_path, output_path)
