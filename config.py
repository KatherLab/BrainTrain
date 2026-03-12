"""
Configuration file for brain MRI classification training

This module handles all configuration settings including:
- Command-line argument parsing
- Data paths and cohort settings
- Model architecture parameters
- Training hyperparameter
- Test outputs and scores
- Explainability and visualization settings

Usage:
    python train.py -c age -m sfcn -g cuda:0
"""
import argparse
import os
from pathlib import Path
from typing import Optional, List, Dict

# ============================================================================
# COMMAND-LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='Brain MRI Classification Training')
parser.add_argument('-c', '--column', type=str, default=None,
                    help='Target column for training (default: label)')
parser.add_argument('-m', '--mode', type=str, default=None,
                    choices=['sfcn', 'dense', 'swin', 'linear', 'ssl-finetuned', 'lora'],
                    help='Training mode (default: lora)')
parser.add_argument('-g', '--gpu', type=str, default=None,
                    help='GPU device (e.g., "cuda:0", default: cuda:0)')
args = parser.parse_args()

# ============================================================================
# BASIC SETTINGS
# ============================================================================

# Task Configuration
COLUMN_NAME = args.column if args.column else 'label'
TRAINING_MODE = args.mode if args.mode else 'sfcn'
TASK = 'regression' 

# Cohort Configuration
TRAIN_COHORT = 'ukb'
TEST_COHORT = 'dlbs'
CSV_NAME_TRAIN = 'demographics'
CSV_NAME_TEST = 'demographics'

# Image Parameters
IMG_SIZE = 96
N_CHANNELS = 1
N_CLASSES = 2
N_CLASSES_EXPLICIT = False
LABEL_MAP_PATH: Optional[str] = None

# ============================================================================
# DATA PATHS
# ============================================================================

# CSV Files
CSV_FULL = f'../data/{TRAIN_COHORT}/{CSV_NAME_TRAIN}.csv'
CSV_TRAIN = f'../data/{TRAIN_COHORT}/train/{CSV_NAME_TRAIN}.csv'
CSV_VAL = f'../data/{TRAIN_COHORT}/val/{CSV_NAME_TRAIN}.csv'
CSV_TEST = f'../data/{TEST_COHORT}/test/{CSV_NAME_TEST}.csv'

# Image Directories
TENSOR_DIR = f'../images/{TRAIN_COHORT}/npy{IMG_SIZE}'
TENSOR_DIR_TEST = f'../images/{TEST_COHORT}/npy{IMG_SIZE}'

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
# Basic Training Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
MIN_LR = 1e-6
NUM_WORKERS = 8
DEVICE = args.gpu if args.gpu else "cuda:0"
SEED = 42
NROWS: Optional[int] = 100  # Set to int for subset, None for all data
GRAD_ACCUM_STEPS = 1
USE_AMP = True
SCHEDULER_TYPE = 'onecycle'  # Options: 'plateau', 'cosine', 'onecycle', 'none'
RESUME_CHECKPOINT: Optional[str] = None
RESUME_RESET_LR = False

# Learning Rate Finder
USE_LR_FINDER = True

# Early Stopping
PATIENCE = 20

# Learning Rate Scheduler (ReduceLROnPlateau)
SCHEDULER_MODE = 'min'  # 'min' for loss, 'max' for accuracy
SCHEDULER_FACTOR = 0.5  # Multiply LR by this factor when reducing
SCHEDULER_PATIENCE = 3  # Number of epochs with no improvement

# ============================================================================
# OUTPUT PATHS
# ============================================================================

# Experiment Naming
# Safely construct experiment name
EXPERIMENT_NAME = f"{COLUMN_NAME}_b{BATCH_SIZE}_lr{LEARNING_RATE}_e{NUM_EPOCHS}_im{IMG_SIZE}"
#EXPERIMENT_NAME = f"{COLUMN_NAME}_b{BATCH_SIZE}_lr{LEARNING_RATE}"

# Output Directories
MODEL_DIR = '../models'
SCORES_DIR = '../scores'
LOG_DIR = '../logs'
EVALUATION_DIR = '../evaluations'
EXPLAINABILITY_DIR = '../explainability'

# Additional Options
KAPLAN_MEIER = False

# Bias Correction for Regression (Age Prediction)
# If True: Bias correction model is fitted on validation set during training
# and applied to test set. This prevents information leakage.
# If False: No bias correction is applied
APPLY_BIAS_CORRECTION = True
BIAS_CORRECTION_COEFFICIENTS_PATH = f'{SCORES_DIR}/{TRAINING_MODE}/val/bias_coeff_{EXPERIMENT_NAME}.json'

# ============================================================================
# EXPLAINABILITY & HEATMAP CONFIGURATION
# ============================================================================

# Visualization Mode
HEATMAP_MODE = 'top_individual'  # Options: 'single', 'average', 'top_individual'
HEATMAP_TOP_N = 5

# Attention Method
ATTENTION_METHOD = 'saliency'  # Options: 'saliency', 'gradcam'
ATTENTION_MODE = 'magnitude'  # Options: 'magnitude', 'signed'
if TASK == 'regression':
    ATTENTION_TARGET = 'output'  # Options: 'output', 'loss'
else:
    ATTENTION_TARGET = 'pred'  # Options: 'logit_diff', 'pred', 'target_class', 'loss'
ATTENTION_CLASS_IDX: Optional[int] = None

# Swin Transformer parameters (override auto selection)
SWIN_PATCH_SIZE = [4, 4, 4]
SWIN_WINDOW_SIZE = [9, 9, 9]

# Brain Atlas Configuration
ATLAS_TYPE = 'AAL'  # Automated Anatomical Labeling
ATLAS_PATH = 'utils/aal3_resampled_96.nii.gz'
N_REGIONS = None  # Number of top regions to analyze

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

# Cohort Configuration
PREPROCESS_COHORT = 'ukb'
REGISTRATION_TYPE = 'Affine'  # Registration algorithm type
CROP_SIZE = 180  # Size before downsampling
PREPROCESS_IMG_SIZE = 180  # Final image size
PREPROCESS_START = 'all'

# Template and Tools
TEMPLATE_PATH = '../path/to/your/template'
DCM2NIIX = '../path/to/your/dcm2niix'

# Processing Directories
DCM_FOLDER = f'../images/{PREPROCESS_COHORT}/dcm_raw/'
INPUT_FOLDER = f'../images/{PREPROCESS_COHORT}/nifti_raw/'
N4_FOLDER = f'../images/{PREPROCESS_COHORT}/nifti_n4/'
REG_FOLDER = f'../images/{PREPROCESS_COHORT}/nifti_reg_{REGISTRATION_TYPE}/'
DESKULL_FOLDER = f'../images/{PREPROCESS_COHORT}/nifti_deskull_{REGISTRATION_TYPE}/'
NPY_FOLDER = f'../images/{PREPROCESS_COHORT}/npy{PREPROCESS_IMG_SIZE}/'

# Processing Parameters
N4_PROCESSES = 4  # Number of parallel processes for N4 bias correction
REG_PROCESSES = 4  # Number of parallel processes for registration
GPU_ID = 0  # GPU device ID for processing

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_config_summary():
    """Print a summary of key configuration settings."""
    print("=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Training Mode: {TRAINING_MODE}")
    print(f"Target Column: {COLUMN_NAME}")
    print(f"Train Cohort: {TRAIN_COHORT}")
    print(f"Test Cohort: {TEST_COHORT}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("=" * 70)

def create_output_directories():
    """Create all necessary output directories."""
    dirs_to_create = [
        MODEL_DIR,
        SCORES_DIR,
        LOG_DIR,
        EVALUATION_DIR,
        EXPLAINABILITY_DIR,
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
