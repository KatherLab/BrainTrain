"""
Simple configuration file for brain MRI classification training
"""
import argparse
import os

parser = argparse.ArgumentParser(description='run training')
parser.add_argument('-c', '--column', type=str, default=None, help='Select one target column for training')
parser.add_argument('-m', '--mode', type=str, default=None, help='Select one of sfcn, dense, linear, ssl-finetuned, lora')
parser.add_argument('-g', '--gpu', type=str, default=None, help='Select GPU device (e.g., "cuda:0")')
args = parser.parse_args()

# ============================================================================
# BASIC SETTINGS
# ============================================================================

COLUMN_NAME = args.column if args.column else 'label'
CSV_NAME = 'ftd-cn' 
CSV_NAME_TEST = 'ftd-cn'
TRAINING_MODE = args.mode if args.mode else 'lora'  # Options: 'sfcn', 'dense', 'linear', 'ssl-finetuned', 'lora'
TASK = 'classification'

# ============================================================================
# DATA PATHS
# ============================================================================

TRAIN_COHORT = 'nifd-m0'
TEST_COHORT = 'BrainLat'
CSV_FULL = f'../data/{TRAIN_COHORT}/{CSV_NAME}.csv'
CSV_TRAIN = f'../data/{TRAIN_COHORT}/train/{CSV_NAME}.csv'
CSV_VAL = f'../data/{TRAIN_COHORT}/val/{CSV_NAME}.csv'
CSV_TEST = f'../data/{TEST_COHORT}/test/{CSV_NAME_TEST}.csv'
TENSOR_DIR = f'../images/{TRAIN_COHORT}/npy96'
TENSOR_DIR_TEST = f'../images/{TEST_COHORT}/npy96'

# ============================================================================
# MODEL SETTINGS
# ============================================================================
IMG_SIZE = 96
N_CHANNELS = 1
N_CLASSES = 2

# LoRA Parameters
LORA_RANK = 16
LORA_ALPHA = 64
LORA_TARGET_MODULES = ["feature_extractor.conv_1.0"]

# SSL Pretrained Model
SSL_COHORT = 'ukb'
SSL_BATCH_SIZE = 16
SSL_EPOCHS = 1000
PRETRAINED_MODEL = (f'../models/ssl/sfcne/{SSL_COHORT}/{SSL_COHORT}{IMG_SIZE}/final_model_b{SSL_BATCH_SIZE}_e{SSL_EPOCHS}.pt')


# ============================================================================
# TRAINING SETTINGS
# ============================================================================
BATCH_SIZE = 16
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
NUM_WORKERS = 8
DEVICE = args.gpu if args.gpu else "cuda:0"
SEED = 42
NROWS = None  # Set to None to use all data, or int for subset

# Early Stopping
PATIENCE = 20

# Learning Rate Scheduler
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

USE_LR_FINDER = True

# ============================================================================
# OUTPUT PATHS
# ============================================================================
# Experiment name
EXPERIMENT_NAME = f"{CSV_NAME}_b{BATCH_SIZE}_lr{LEARNING_RATE}_ssl-{SSL_COHORT}_lora-{LORA_TARGET_MODULES}"
#EXPERIMENT_NAME = f"{CSV_NAME}_b{BATCH_SIZE}_lr{LEARNING_RATE}_lora{LORA_TARGET_MODULES}"
# Output directories
MODEL_DIR = f'../models'
SCORES_DIR = f'../scores'
LOG_DIR = f'../logs'
EVALUATION_DIR = f'../evaluations'
EXPLAINABILITY_DIR = f"../explainability"
KAPLAN_MEIER = False



# ============================================================================
# HEATMAP CONFIGURATION 
# ============================================================================
HEATMAP_MODE = 'top_individual'  # Options: 'single', 'average', 'top_individual'
HEATMAP_TOP_N = 5
ATTENTION_METHOD = 'saliency'  # Options: 'saliency', 'gradcam'
ATTENTION_MODE = 'magnitude'  # Options: 'magnitude', 'signed'
ATTENTION_TARGET = 'logit_diff'  # Options: 'logit_diff', 'pred', 'target_class'
ATTENTION_CLASS_IDX = None
ATLAS_TYPE = 'AAL'
ATLAS_PATH = 'utils/aal3_resampled_96.nii.gz'
N_REGIONS = 100


# ============================================================================
# SSL CONFIGURATION 
# ============================================================================

SEED_SSL = 42
COHORT_SSL = 'ukb'
MAX_EPOCHS_SSL = 1000
VAL_INTERVAL_SSL = 1
BATCH_SIZE_SSL = 16
LEARNING_RATE_SSL = 1e-1
NUM_WORKERS = 8
MODEL_TYPE_SSL = 'sfcne'
MODEL_NAME_SSL = f'final_model_b{BATCH_SIZE_SSL}_e{MAX_EPOCHS_SSL}.pt'
# Loss Parameters
CONTRASTIVE_TEMPERATURE = 0.05
# Early Stopping
PATIENCE_SSL = 10
SCHEDULER_FACTOR_SSL = 0.5
SCHEDULER_PATIENCE_SSL = 5

# ============================================
# DIMENSIONALITY REDUCTION SETTINGS
# ============================================

# Choose reduction method: 'umap', 'tsne', or 'pca'
REDUCTION_METHOD = 'umap'  # Options: 'umap', 'tsne', 'pca'
# UMAP-specific parameters
N_NEIGHBORS = 6        # Number of neighbors (UMAP only)
MIN_DIST = 1       # Minimum distance (UMAP only)
# t-SNE-specific parameters 
PERPLEXITY = 100        # Perplexity (t-SNE only, typically 5-50)
# General parameters
RANDOM_STATE = 42       # Random seed for reproducibility


# ============================================
# REPRESENTATIONS SETTINGS
# ============================================
# Data
CSV_NAME = 'ms-cn_balanced'
COHORT_EXTRACT = 'BrainLat'
IMG_SIZE = 96

POINT_SIZE = 600
TRANSPARENCY = 0.6
FONTSIZE_MAX = 30
FONTSIZE_MIN = 40
# Define a fixed color mapping for all diagnostic categories
DIAGNOSIS_COLORS = {
    'CN': 'lightgrey',    # Grey (controls)
    'AD': '#FFA500',    # Orange
    'FTD': '#6495ED',   # Cornflower blue
    'PD': '#2E8B57',    # Sea green
    'MS': '#F48FB1',    # Light pink
    'BV': '#9370DB',    # Medium purple (bvFTD)
    'SV': '#20B2AA',    # Light sea green (svPPA)
    'PNFA': '#FF6347',  # Tomato red (nfvPPA)
    # Add other categories as needed
}

# ============================================
# CONFIGURE PATHS
# ============================================
# File Paths (CHANGE THESE TO YOUR PATHS)
JSON_PATH = f'../data/ssl_data/ssl-{COHORT_SSL}.json'
LOG_DIR_SSL = f"../logs/ssl/{MODEL_TYPE_SSL}/{COHORT_SSL}/{COHORT_SSL}{IMG_SIZE}/"
MODEL_DIR_SSL = f"../models/ssl/{MODEL_TYPE_SSL}/{COHORT_SSL}/{COHORT_SSL}{IMG_SIZE}/"
IMAGES_EXT_DIR = f'../images/{COHORT_EXTRACT}/npy96/'
FEATURES_EXT_DIR = f"../features/{COHORT_EXTRACT}/{MODEL_TYPE_SSL}/"
VIZ_DIR = f"../representations/{COHORT_EXTRACT}/{MODEL_TYPE_SSL}/ssl-{COHORT_SSL}/{MODEL_NAME_SSL}/{REDUCTION_METHOD}/"
DATA_PATH = f'../data/{COHORT_EXTRACT}/{CSV_NAME}.csv'

# ============================================
# PREPROCESSING SETTINGS
# ============================================
# === Cohort Settings ===
cohort = 'trial'
reg_type = 'Affine'
crop_size = 180 
img_size = 96  # Add your image size here

# === Paths ===
template_path = '../images/templates/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
DCM2NIIX = '../.venv/bin/dcm2niix'

dcm_folder = f'../images/{cohort}/dcm_raw/'
input_folder = f'../images/{cohort}/nifti_raw/'
n4_folder = f'../images/{cohort}/nifti_n4/'
reg_folder = f'../images/{cohort}/nifti_reg_{reg_type}/'
deskull_folder = f'../images/{cohort}/nifti_deskull_{reg_type}/'
npy_folder = f'../images/{cohort}/npy{img_size}/'

# === Processing Parameters ===
N4_PROCESSES = 4
REG_PROCESSES = 4
GPU_ID = 0
