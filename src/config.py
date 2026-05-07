from pathlib import Path

# ==========================================
# 📂 DIRECTORY STRUCTURE
# ==========================================
BASE_PATH = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_PATH
DATA_PROCESSED = BASE_PATH / "data" / "processed"

# Sub-directories
EPOCHS_DIR = DATA_PROCESSED / "master_epochs"
REFINED_DIR = DATA_PROCESSED / "refined_master"
TENSOR_DIR = DATA_PROCESSED / "connectivity"
OUTPUTS_DIR = BASE_PATH / "outputs" / "eda"

# Create directories if not exist
for d in [EPOCHS_DIR, REFINED_DIR, TENSOR_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================
# 🧠 EEG CONFIGURATION (Ozdemir 2017)
# ==========================================
SFREQ = 128  # Hz
TMIN, TMAX = -1.0, 1.0
BASELINE = (-0.6, -0.4)

# 30 EEG Channels for high spatial resolution
EEG_CHANNELS = [
    'Fp1', 'F3', 'F7', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz',
    'Fp2', 'Fz', 'F4', 'F8', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2'
]

EOG_CHANNELS = ['HEOG_left', 'HEOG_right', 'VEOG_lower']

# Montage
MONTAGE_NAME = "standard_1005"

# ==========================================
# ⚡ PROCESSING PARAMETERS
# ==========================================
FILTER_LOW = 0.1
FILTER_HIGH = 30.0
THETA_BAND = (4.0, 8.0)  # Hz

# ICA
ICA_COMPONENTS = 10
ICA_METHOD = 'infomax'

# Thresholds
MIN_INCORRECT_TRIALS = 15

# ==========================================
# 📊 OUTPUT FILENAMES
# ==========================================
TENSOR_CORRECT_FILE = TENSOR_DIR / "tensor_correct_4d.npy"
TENSOR_INCORRECT_FILE = TENSOR_DIR / "tensor_incorrect_4d.npy"
REPORT_ETL = DATA_PROCESSED / "master_preprocessing_report.csv"
