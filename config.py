# -*- coding: utf-8 -*-
import os
import torch
import torch.backends.cudnn as cudnn
import optuna

# --- Project Path (MODIFY IF NEEDED) ---
# Base directory for the project
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Assumes config.py is in the root
# Local directory to store outputs (models, results, etc.) relative to project root
OUTPUT_DIR_NAME = 'st311_project_output'
PROJECT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, OUTPUT_DIR_NAME)
os.makedirs(PROJECT_OUTPUT_PATH, exist_ok=True)
# print(f"Project outputs will be saved in: {PROJECT_OUTPUT_PATH}")

# --- Path to Downloaded Kaggle Dataset ---
# This will be set by main.py after attempting the download.
# If download fails, manually set this path to your dataset location.
# Example: DATASET_BASE_PATH = '/path/to/your/downloaded/st311-tennis/'
DATASET_BASE_PATH = "C:/Users/josep/.cache/kagglehub/datasets/joekinder/st311-tennis/versions/1/"

# --- Core Parameters ---
IMG_HEIGHT, IMG_WIDTH = 224, 224 # Input image dimensions

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Enable CuDNN benchmark mode for potentially faster convolutions if input sizes don't vary much
# Disable if input sizes change a lot (less common in this type of task)
cudnn.benchmark = True
# print(f"Using device: {DEVICE}")

# --- Default Hyperparameters (can be overridden by loaded best params) ---
# Data Prep
DEFAULT_N_FRAMES_WEIGHTING = 7 # Frame window for CNN1 weight calculation
DEFAULT_WEIGHT_DECAY = 0.3   # Decay factor for CNN1 frame weights
DEFAULT_BALANCE_RATIO = 4    # Ratio of non-hit frames to weighted frames for CNN1 dataset

# CNN1 Training
DEFAULT_CNN1_BATCH_SIZE = 32 # Increased default, tune based on GPU memory
DEFAULT_CNN1_LR = 5e-4
DEFAULT_CNN1_FILTERS = (32, 64, 128) # Initial default arch
DEFAULT_CNN1_FC_SIZE = 512           # Initial default arch
DEFAULT_CNN1_DROPOUT = 0.5

# CNN2 Training
DEFAULT_N_FRAMES_SEQUENCE_CNN2 = 7 # Input sequence length for CNN2
DEFAULT_CNN2_BATCH_SIZE = 16 # Often needs smaller batch size due to larger input tensor
DEFAULT_CNN2_LR = 5e-5
# Derived - DO NOT EDIT MANUALLY, updated based on N_FRAMES_SEQUENCE_CNN2
DEFAULT_CNN2_INPUT_CHANNELS = DEFAULT_N_FRAMES_SEQUENCE_CNN2 * 3

# Training Control
DEFAULT_FINAL_EPOCHS = 150      # Max epochs for the *final* training run
DEFAULT_EARLY_STOPPING_PATIENCE = 15 # Patience for final training early stopping
DEFAULT_MIN_IMPROVEMENT = 1e-5 # Min improvement threshold for early stopping

# Data Loader Efficiency
# Adjust based on your system's cores and performance profiling
# Start with 2 or 4, 0 means loading in the main process (can be slow)
NUM_WORKERS = 4 if DEVICE.type == 'cuda' else 0 # Often benefits from more workers with GPU
PIN_MEMORY = True if DEVICE.type == 'cuda' else False # Speeds up CPU->GPU transfer

# Grid Search Control
GRID_SEARCH_TUNING_EPOCHS = 5 # Number of epochs for *each* grid search trial
GRID_SEARCH_ARCHITECTURE_CANDIDATES = 15 # Max architectures to try for CNN1
# Grid search parameters (to be defined in grid_search.py)

# Landing Spot Coordinate Normalization
DOUBLES_COURT_WIDTH_M = 10.97
HALF_COURT_LENGTH_M = 11.89 # Approx distance from net to baseline

# --- Linear Weighting Defaults (for Standard CNN1 Training) ---
DEFAULT_LINEAR_N_FRAMES_WEIGHTING = 7 # Renamed
DEFAULT_LINEAR_WEIGHT_DECAY = 0.3   # Renamed

# --- CNN1 Defaults ---
DEFAULT_BALANCE_RATIO = 4
DEFAULT_CNN1_BATCH_SIZE = 32
DEFAULT_CNN1_LR = 5e-4
DEFAULT_CNN1_FILTERS = (32, 64, 128)
DEFAULT_CNN1_FC_SIZE = 512
DEFAULT_CNN1_DROPOUT = 0.5

# --- CNN2 Defaults ---
DEFAULT_N_FRAMES_SEQUENCE_CNN2 = 7 # Used for standard CNN2 training/eval input seq len
DEFAULT_CNN2_BATCH_SIZE = 16
DEFAULT_CNN2_LR = 5e-5
DEFAULT_CNN2_INPUT_CHANNELS = DEFAULT_N_FRAMES_SEQUENCE_CNN2 * 3

# --- Training Control ---
DEFAULT_FINAL_EPOCHS = 150
DEFAULT_EARLY_STOPPING_PATIENCE = 15
DEFAULT_MIN_IMPROVEMENT = 1e-5
NUM_WORKERS = 4 if DEVICE.type == 'cuda' else 0
PIN_MEMORY = True if DEVICE.type == 'cuda' else False

# --- Grid Search Control ---
GRID_SEARCH_TUNING_EPOCHS = 5
GRID_SEARCH_ARCHITECTURE_CANDIDATES = 15

# --- Bayesian Optimization Control (Expanded) ---
BAYESIAN_OPT_N_TRIALS = 50 # Increase trials for more params
BAYESIAN_OPT_TUNING_EPOCHS = 5
# Define search space for h(x) parameters (R1/R2 are INTEGERS)
BAYESIAN_PARAM_RANGES = {
    "R1": (1, 10),     # Max frames AFTER predicted hit (Integer)
    "R2": (1, 10),     # Max frames BEFORE predicted hit (Integer)
    "N": (0.01, 2.0),  # Suggest floats, avoid N=0
    "D": (0.01, 2.0),  # Suggest floats, avoid D=0
    "M1": (0.0, 0.5),  # Suggest floats (keep small to avoid exp explosion?)
    "M2": (0.0, 0.5)   # Suggest floats (keep small to avoid exp explosion?)
}
# Store the best found Bayesian params globally after optimization
BEST_BAYESIAN_PARAMS = None

# --- Joint Training Control ---
DEFAULT_JOINT_LR = 1e-5
DEFAULT_JOINT_BATCH_SIZE = 16
DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT = 0.1
# Fixed length for the context window in JointPredictionDataset
JOINT_DATASET_CONTEXT_FRAMES = 21 # Must be >= max(R1+R2+1). Needs to be odd.

# --- Coordinate Normalization ---
DOUBLES_COURT_WIDTH_M = 10.97
HALF_COURT_LENGTH_M = 11.89

# --- Store optimized R1/R2 for joint training loop ---
# These will be populated after Bayesian optimization
OPTIMIZED_R1_INT = None
OPTIMIZED_R2_INT = None