import numpy as np
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def assemble_tensors(input_path, output_path):
    # Find all correct and incorrect files
    correct_files = sorted(list(input_path.glob("*_correct_plv.npy")))
    incorrect_files = sorted(list(input_path.glob("*_incorrect_plv.npy")))
    
    if not correct_files:
        logger.error("No PLV files found to assemble.")
        return

    # Load first file to get dimensions
    sample = np.load(correct_files[0])
    N1, N2, T = sample.shape
    S = 40 # Total subjects, but we should use actual found count
    
    def stack_files(files, name):
        s_count = len(files)
        # 4D Tensor: (N, N, T, S)
        tensor = np.zeros((N1, N2, T, s_count), dtype=np.float32)
        for i, f in enumerate(files):
            data = np.load(f)
            # Ensure shape matches
            if data.shape == (N1, N2, T):
                tensor[:, :, :, i] = data
            else:
                logger.warning(f"Shape mismatch in {f}: {data.shape} vs {(N1, N2, T)}")
        
        save_path = output_path / f"{name}_4d_tensor.npy"
        # Using memmap-like saving or just save
        np.save(save_path, tensor)
        logger.info(f"Saved {name} 4D tensor with shape {tensor.shape} to {save_path}")

    stack_files(correct_files, "correct")
    stack_files(incorrect_files, "incorrect")

if __name__ == "__main__":
    base_path = Path("/home/chinh303/Downloads/ERN Raw Data BIDS-Compatible")
    input_path = base_path / "outputs" / "connectivity"
    output_path = base_path / "outputs" / "tensors"
    output_path.mkdir(parents=True, exist_ok=True)
    
    assemble_tensors(input_path, output_path)
