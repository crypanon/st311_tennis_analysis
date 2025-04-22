# -*- coding: utf-8 -*-
import os
import warnings
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse # For command-line arguments
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math # For checking float validity
import re # Added for fallback dir finding
import json # Added for saving/loading json directly if needed

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.data')
warnings.filterwarnings('ignore', category=FutureWarning) # Often from pandas/numpy via dependencies

# --- Project Imports ---
import config # Load configuration first
from utils import load_json_params, plot_training_history, setup_kaggle_downloader
from data_utils import (load_metadata, apply_linear_weighting_to_df, # Use linear
                        apply_bayesian_weighting_to_df, # Use bayesian
                        balance_and_split_data, load_landing_data,
                        get_sequences_for_cnn2, get_long_context_sequences, # Use long context
                        split_sequences)
# Add LandingPointCNNParam to imports if needed by grid_search
from datasets import TennisFrameDataset, BallLandingDataset, JointPredictionDataset # Add Joint
from models import (HitFrameRegressorFinal, LandingPointCNN, LandingPointCNNParam) # Add CNN2 Param
from training import train_model, evaluate_model, train_joint_model # Add Joint
# Add run_cnn2_arch_search to grid_search imports
from grid_search import (run_cnn1_arch_search, run_cnn1_trainhp_search,
                         run_cnn2_dataprep_search, run_cnn2_trainhp_search, run_cnn2_arch_search) # Add CNN2 arch search
from bayesian_optimizer import run_bayesian_optimization # NEW Import
# Import prediction function that accepts R1/R2
from prediction import (load_final_cnn1_model, load_final_cnn2_model,
                        predict_hit_and_landing, denormalize_coordinates) # Updated prediction

# Set seed for reproducibility (optional)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def main(args):
    """Main execution function."""
    start_time_main = time.time()

    # --- 1. Dataset Setup ---
    dataset_base_path = config.DATASET_BASE_PATH
    if not dataset_base_path or not os.path.exists(dataset_base_path):
        if args.download_data:
            print("\nAttempting to download dataset via Kaggle Hub...")
            dataset_base_path = setup_kaggle_downloader()
            if dataset_base_path:
                 print(f"Using downloaded dataset path: {dataset_base_path}")
                 config.DATASET_BASE_PATH = dataset_base_path # Update config if successful
            else:
                 print("Automatic download failed. Please set DATASET_BASE_PATH in config.py manually.")
                 return
        else:
            print("\nDataset path not set or invalid in config.py.")
            if args.run_grid_search or args.run_bayesian_opt or args.run_final_training or args.run_joint_training:
                 print("Exiting as data is required for optimization/training.")
                 return
            elif args.run_prediction or args.run_evaluation:
                 print("Proceeding without dataset path (assuming models/results exist).")
                 dataset_base_path = None
            else:
                 return

    # --- 2. Load Metadata (if path is valid) ---
    df_full = None
    if dataset_base_path and os.path.exists(dataset_base_path):
        csv_path = os.path.join(dataset_base_path, 'Frames/hit_frames.csv')
        try:
            df_full = load_metadata(csv_path, dataset_base_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            if args.run_grid_search or args.run_bayesian_opt or args.run_final_training or args.run_joint_training: return
        except Exception as e:
            print(f"An unexpected error occurred during metadata loading: {e}")
            if args.run_grid_search or args.run_bayesian_opt or args.run_final_training or args.run_joint_training: return
    elif args.run_grid_search or args.run_bayesian_opt or args.run_final_training or args.run_joint_training:
         print("Error: Cannot proceed with optimization/training without a valid dataset path.")
         return


    # --- 3. Initialize Hyperparameters ---
    cnn1_weighting_params = None # For h(x) params (R1, R2, N, D, M1, M2)
    cnn1_arch_params = {'filters': config.DEFAULT_CNN1_FILTERS, 'fc_size': config.DEFAULT_CNN1_FC_SIZE, 'dropout': config.DEFAULT_CNN1_DROPOUT}
    cnn1_trainhp_params = {'learning_rate': config.DEFAULT_CNN1_LR, 'batch_size': config.DEFAULT_CNN1_BATCH_SIZE}
    # Add CNN2 Arch params init
    cnn2_arch_params = {'conv_filters': config.DEFAULT_CNN2_CONV_FILTERS, 'fc_sizes': config.DEFAULT_CNN2_FC_SIZES, 'dropout': config.DEFAULT_CNN2_DROPOUT}
    cnn2_dataprep_params = {'n_frames_sequence_cnn2': config.DEFAULT_N_FRAMES_SEQUENCE_CNN2} # For standard CNN2
    cnn2_trainhp_params = {'learning_rate': config.DEFAULT_CNN2_LR, 'batch_size': config.DEFAULT_CNN2_BATCH_SIZE}


    # --- 4. Optional Searches (Grid for Arch/HP, BayesOpt for Weighting) ---
    final_cnn1_splits = None # Uses linear weights if only standard training
    final_cnn2_splits = None # Standard CNN2 splits
    final_joint_splits = None # Splits for JointPredictionDataset
    cnn1_balanced_df_final = None # To store balanced DF corresponding to best weights
    landing_df_indexed = None # To store loaded landing data
    df_processed_linear = None # Store DF with linear weights if created

    # Define paths to check for existence
    arch_json_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_architecture.json')
    bayes_weights_json_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_bayesian_weights.json')
    cnn1_hp_json_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_training_hp.json')
    cnn2_arch_json_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_architecture.json') # Added
    cnn2_dp_json_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_dataprep.json')
    cnn2_hp_json_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_training_hp.json')

    run_any_optimization = args.run_grid_search or args.run_bayesian_opt

    if run_any_optimization:
        if df_full is None:
            print("ERROR: Cannot run search/optimization without loaded metadata (df_full).")
            return

        print("\n" + "#"*20 + " Starting Hyperparameter Optimization Phase " + "#"*20)

        # --- CNN1 Architecture Search (Grid Search) ---
        # Run only if requested AND file doesn't exist
        if args.run_grid_search and not os.path.exists(arch_json_path):
             print("\nRunning CNN1 Architecture Search...")
             print("Using LINEAR weighting for CNN1 Arch search prep...")
             df_processed_linear = apply_linear_weighting_to_df(
                 df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY
             )
             initial_cnn1_splits = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
             best_filters, best_fc, best_dropout = run_cnn1_arch_search(df_full, initial_cnn1_splits, config.DEVICE)
             # Use correct key 'dropout_rate' if possible, handle 'dropout' for legacy
             cnn1_arch_params = {'filters': best_filters, 'fc_size': best_fc, 'dropout_rate': best_dropout}
             print("CNN1 Architecture Search Complete.")
        else:
             loaded_arch = load_json_params(arch_json_path, "CNN1 Arch")
             if loaded_arch: cnn1_arch_params = loaded_arch
             else: print("WARN: CNN1 Arch params file not found, using defaults.");


        # --- CNN1 h(x) Weighting Parameter Search (Bayesian Optimization) ---
        # Run only if requested AND file doesn't exist
        if args.run_bayesian_opt and not os.path.exists(bayes_weights_json_path):
             print("\nRunning Bayesian Optimization for h(x) Weighting...")
             # Ensure arch params are passed correctly (need tuple values)
             cnn1_arch_tuple = (tuple(cnn1_arch_params.get('filters', config.DEFAULT_CNN1_FILTERS)),
                                cnn1_arch_params.get('fc_size', config.DEFAULT_CNN1_FC_SIZE),
                                cnn1_arch_params.get('dropout', cnn1_arch_params.get('dropout_rate', config.DEFAULT_CNN1_DROPOUT)))
             cnn1_weighting_params = run_bayesian_optimization(df_full, cnn1_arch_tuple, config.DEVICE)
             # R1/R2 integers are stored in config.OPTIMIZED_R1_INT/R2_INT by the function
             print("Bayesian Optimization Complete.")
        else:
             loaded_weights = load_json_params(bayes_weights_json_path, "CNN1 Bayesian Weights")
             if loaded_weights:
                  cnn1_weighting_params = loaded_weights
                  config.OPTIMIZED_R1_INT = int(round(loaded_weights.get('R1', 5)))
                  config.OPTIMIZED_R2_INT = int(round(loaded_weights.get('R2', 5)))
                  print(f"Loaded Optimized R1 (int): {config.OPTIMIZED_R1_INT}, R2 (int): {config.OPTIMIZED_R2_INT}")
             else:
                  print("Warning: BayesOpt weight params file not found.")
                  if args.run_joint_training: # Critical for joint training
                      print("Cannot run Joint Training without Bayesian weights. Exiting.")
                      return
                  else: # Use fallback for other steps if possible (though suboptimal)
                      print("Using fallback Bayesian parameters.")
                      cnn1_weighting_params = {"R1": 5.0, "R2": 5.0, "N": 0.5, "D": 0.5, "M1": 0.1, "M2": 0.1}
                      config.OPTIMIZED_R1_INT = 5
                      config.OPTIMIZED_R2_INT = 5


        # --- CNN1 Training HP Search (Grid Search) ---
        # Run only if requested AND file doesn't exist
        if args.run_grid_search and not os.path.exists(cnn1_hp_json_path):
             print("\nRunning CNN1 Training HP Search...")
             # Uses LINEAR weighting for standard CNN1 HP search
             print("Using LINEAR weights for CNN1 HP Search prep...")
             if df_processed_linear is None: # Create if not already done
                 df_processed_linear = apply_linear_weighting_to_df(
                    df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY
                 )
             splits_for_hp_search = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
             # Pass arch params as tuple
             cnn1_arch_tuple = (tuple(cnn1_arch_params.get('filters', config.DEFAULT_CNN1_FILTERS)),
                                cnn1_arch_params.get('fc_size', config.DEFAULT_CNN1_FC_SIZE),
                                cnn1_arch_params.get('dropout', cnn1_arch_params.get('dropout_rate', config.DEFAULT_CNN1_DROPOUT)))
             best_lr_cnn1, best_bs_cnn1 = run_cnn1_trainhp_search(splits_for_hp_search, cnn1_arch_tuple, config.DEVICE)
             cnn1_trainhp_params = {'learning_rate': best_lr_cnn1, 'batch_size': best_bs_cnn1}
             print("CNN1 Training HP Search Complete.")
        else:
             loaded_hp1 = load_json_params(cnn1_hp_json_path, "CNN1 TrainHP")
             if loaded_hp1: cnn1_trainhp_params = loaded_hp1
             else: print("WARN: CNN1 Train HP params file not found, using defaults.");


        # --- CNN2 Searches (Seq Len, Arch, HP for standard CNN2) ---
        run_any_cnn2_search = args.run_grid_search and (
            not os.path.exists(cnn2_dp_json_path) or
            not os.path.exists(cnn2_hp_json_path) or
            not os.path.exists(cnn2_arch_json_path) # Check arch file too
        )
        if run_any_cnn2_search:
            print("\nRunning Standard CNN2 Searches (Seq Len, Arch, HP)...")
            # Load landing data needed for all CNN2 searches
            if landing_df_indexed is None:
                 if dataset_base_path is None: print("ERR: Dataset path needed for landing data."); return
                 try: landing_df_indexed = load_landing_data(dataset_base_path)
                 except Exception as e: print(f"ERROR loading landing data: {e}"); return

            # Prep data using linear weights (consistent standard setup)
            if df_processed_linear is None:
                df_processed_linear = apply_linear_weighting_to_df(df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY)
            # Need splits based on linear weights
            splits_for_cnn2_search = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
            cnn1_balanced_df_for_cnn2_search = splits_for_cnn2_search[0]

            # a) CNN2 Seq Len Search
            if not os.path.exists(cnn2_dp_json_path):
                 print("\nRunning CNN2 Sequence Length Search...")
                 best_seq_len, final_cnn2_splits_unpacked = run_cnn2_dataprep_search(cnn1_balanced_df_for_cnn2_search, landing_df_indexed, config.DEVICE)
                 cnn2_dataprep_params = {'n_frames_sequence_cnn2': best_seq_len}
                 # Store these splits as they correspond to the best standard CNN2 seq len
                 final_cnn2_splits = (final_cnn2_splits_unpacked[0], final_cnn2_splits_unpacked[1], final_cnn2_splits_unpacked[2])
                 print("CNN2 Sequence Length Search Complete.")
            else: # Load if exists, still need splits for subsequent searches if they haven't run
                 loaded_dp2 = load_json_params(cnn2_dp_json_path, "CNN2 DataPrep")
                 if loaded_dp2: cnn2_dataprep_params = loaded_dp2
                 else: print("WARN: CNN2 DataPrep file not found, using defaults.");
                 # Recreate splits if needed for subsequent searches
                 if final_cnn2_splits is None and (not os.path.exists(cnn2_arch_json_path) or not os.path.exists(cnn2_hp_json_path)):
                     print("Recreating standard CNN2 splits for Arch/HP search...")
                     current_seq_len = cnn2_dataprep_params['n_frames_sequence_cnn2']
                     # Ensure seq len is odd for get_sequences_for_cnn2
                     if current_seq_len % 2 == 0: current_seq_len +=1
                     final_sequences_cnn2 = get_sequences_for_cnn2(cnn1_balanced_df_for_cnn2_search, landing_df_indexed, current_seq_len)
                     if not final_sequences_cnn2: print("ERR: Failed to recreate CNN2 sequences"); return
                     f_train_seq2, f_val_seq2, f_test_seq2 = split_sequences(final_sequences_cnn2)
                     final_cnn2_splits = (f_train_seq2, f_val_seq2, f_test_seq2)

            # b) CNN2 Architecture Search (NEW)
            if not os.path.exists(cnn2_arch_json_path):
                 if final_cnn2_splits is None: print("ERR: CNN2 splits needed for Arch Search."); return
                 print("\nRunning CNN2 Architecture Search...")
                 current_seq_len = cnn2_dataprep_params['n_frames_sequence_cnn2']
                 # Ensure seq len is odd for consistency with data prep used
                 if current_seq_len % 2 == 0: current_seq_len +=1
                 best_conv_filters, best_fc_sizes, best_dropout_cnn2 = run_cnn2_arch_search(
                     standard_cnn2_splits=final_cnn2_splits,
                     standard_cnn2_seq_len=current_seq_len,
                     device=config.DEVICE
                 )
                 cnn2_arch_params = {'conv_filters': best_conv_filters, 'fc_sizes': best_fc_sizes, 'dropout': best_dropout_cnn2}
                 print("CNN2 Architecture Search Complete.")
            else: # Load if exists
                 loaded_arch2 = load_json_params(cnn2_arch_json_path, "CNN2 Arch")
                 if loaded_arch2: cnn2_arch_params = loaded_arch2
                 else: print("WARN: CNN2 Arch file not found, using defaults.");

            # c) CNN2 Training HP Search
            if not os.path.exists(cnn2_hp_json_path):
                 if final_cnn2_splits is None: print("ERR: CNN2 splits needed for HP Search."); return
                 print("\nRunning CNN2 Training HP Search...")
                 current_seq_len = cnn2_dataprep_params['n_frames_sequence_cnn2']
                 if current_seq_len % 2 == 0: current_seq_len +=1 # Ensure odd
                 best_lr_cnn2, best_bs_cnn2 = run_cnn2_trainhp_search(final_cnn2_splits, current_seq_len, config.DEVICE)
                 cnn2_trainhp_params = {'learning_rate': best_lr_cnn2, 'batch_size': best_bs_cnn2}
                 print("CNN2 Training HP Search Complete.")
            else: # Load if exists
                 loaded_hp2 = load_json_params(cnn2_hp_json_path, "CNN2 TrainHP")
                 if loaded_hp2: cnn2_trainhp_params = loaded_hp2
                 else: print("WARN: CNN2 HP file not found, using defaults.");

        elif run_any_optimization: # if opt was requested but CNN2 files exist
             print("\nSkipping CNN2 Searches (Seq Len, Arch, HP) as results files exist.")
             # Load existing CNN2 params
             loaded_dp2 = load_json_params(cnn2_dp_json_path, "CNN2 DataPrep");
             if loaded_dp2: cnn2_dataprep_params = loaded_dp2
             loaded_arch2 = load_json_params(cnn2_arch_json_path, "CNN2 Arch");
             if loaded_arch2: cnn2_arch_params = loaded_arch2
             loaded_hp2 = load_json_params(cnn2_hp_json_path, "CNN2 TrainHP");
             if loaded_hp2: cnn2_trainhp_params = loaded_hp2

        print("\n" + "#"*20 + " Hyperparameter Optimization Phase Finished " + "#"*20)

    else: # Neither grid search nor bayes opt requested, load all from files
        print("\nSkipping optimization. Loading best parameters from JSON files...")
        loaded_arch = load_json_params(arch_json_path, "CNN1 Arch")
        if loaded_arch: cnn1_arch_params = loaded_arch
        loaded_weights = load_json_params(bayes_weights_json_path, "CNN1 Bayes")
        if loaded_weights:
             cnn1_weighting_params = loaded_weights
             config.OPTIMIZED_R1_INT = int(round(loaded_weights.get('R1', 5)))
             config.OPTIMIZED_R2_INT = int(round(loaded_weights.get('R2', 5)))
             print(f"Loaded Optimized R1 (int): {config.OPTIMIZED_R1_INT}, R2 (int): {config.OPTIMIZED_R2_INT}")
        else:
            print("Warning: Bayesian weight params file not found. Cannot run Joint Training.")
            if args.run_joint_training: return
        loaded_hp1 = load_json_params(cnn1_hp_json_path, "CNN1 HP")
        if loaded_hp1: cnn1_trainhp_params = loaded_hp1
        # Load CNN2 params
        loaded_arch2 = load_json_params(cnn2_arch_json_path, "CNN2 Arch") # Load arch
        if loaded_arch2: cnn2_arch_params = loaded_arch2
        loaded_dp2 = load_json_params(cnn2_dp_json_path, "CNN2 DP")
        if loaded_dp2: cnn2_dataprep_params = loaded_dp2
        loaded_hp2 = load_json_params(cnn2_hp_json_path, "CNN2 HP")
        if loaded_hp2: cnn2_trainhp_params = loaded_hp2


    # --- 5. Final Data Preparation (Standard & Joint) ---
    # Declare loaders here to handle scope if only eval/pred is run
    cnn1_train_loader, cnn1_val_loader, cnn1_test_loader = None, None, None
    cnn2_train_loader, cnn2_val_loader, cnn2_test_loader = None, None, None
    joint_train_loader, joint_val_loader, joint_test_loader = None, None, None

    if args.run_final_training or args.run_joint_training or args.run_evaluation:
        if df_full is None: print("ERROR: df_full needed for data preparation."); return
        # Ensure landing data is loaded if needed for CNN2 or Joint prep
        if landing_df_indexed is None:
            if dataset_base_path is None: print("ERR: Dataset path needed to load landing data."); return
            try: landing_df_indexed = load_landing_data(dataset_base_path)
            except Exception as e: print(f"ERROR loading landing data: {e}"); return

        print("\nPreparing final datasets...")

        # Regenerate CNN1 data splits using LINEAR weights for standard training/eval
        print("Applying LINEAR weights for Standard CNN1 data...")
        if df_processed_linear is None: # Create if not done during searches
             df_processed_linear = apply_linear_weighting_to_df(
                 df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY
             )
        final_cnn1_splits = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
        cnn1_balanced_df_linear = final_cnn1_splits[0] # Store balanced df from linear weights
        _, f_train_p1, f_train_t1, f_val_p1, f_val_t1, f_test_p1, f_test_t1 = final_cnn1_splits
        if not f_train_p1 or not f_val_p1 or not f_test_p1: print("ERROR: Final std CNN1 split failed."); return

        # Prepare standard CNN2 sequences/splits
        try:
            # Use the standard ODD sequence length from CNN2 DP params
            cnn2_seq_len_std = cnn2_dataprep_params.get('n_frames_sequence_cnn2', config.DEFAULT_N_FRAMES_SEQUENCE_CNN2)
            if cnn2_seq_len_std % 2 == 0: cnn2_seq_len_std += 1 # Ensure odd
            val_test_cnn2_seq_len = cnn2_seq_len_std
            print(f"Preparing Standard CNN2 sequences using length: {val_test_cnn2_seq_len}")

            final_sequences_cnn2 = get_sequences_for_cnn2(
                cnn1_balanced_df_linear, landing_df_indexed, val_test_cnn2_seq_len
            )
            if not final_sequences_cnn2: raise ValueError("No std CNN2 sequences.")
            f_train_seq2, f_val_seq2, f_test_seq2 = split_sequences(final_sequences_cnn2)
            # Store the actual splits used
            final_cnn2_splits = (f_train_seq2, f_val_seq2, f_test_seq2)
            if not f_train_seq2 or not f_val_seq2 or not f_test_seq2: raise ValueError("Std CNN2 split failed.")
        except Exception as e: print(f"ERROR prep standard CNN2: {e}"); return

        # Prepare JOINT sequences/splits using BEST Bayesian h(x) weights (only if needed)
        if args.run_joint_training:
            if cnn1_weighting_params is None: print("ERR: Bayes weights needed for joint prep."); return
            try:
                print("Applying final Bayesian weights for joint dataset...")
                df_with_target_weights = apply_bayesian_weighting_to_df(df_full, **cnn1_weighting_params)
                print("Preparing long context sequences...")
                final_sequences_joint = get_long_context_sequences(
                    df_full, # Pass original df containing is_hit_frame
                    landing_df_indexed,
                    df_with_target_weights[['frame_path', 'weight']], # Pass df with target weights
                    context_len=config.JOINT_DATASET_CONTEXT_FRAMES
                )
                if not final_sequences_joint: raise ValueError("No Joint sequences.")
                f_train_seqJ, f_val_seqJ_joint, f_test_seqJ_joint = split_sequences(final_sequences_joint) # Split joint format data
                # Val/Test for joint training will use standard CNN2 val/test sequences/loaders
                f_val_seqJ = final_cnn2_splits[1] # Use standard val set
                f_test_seqJ = final_cnn2_splits[2] # Use standard test set
                final_joint_splits = (f_train_seqJ, f_val_seqJ, f_test_seqJ)
                if not f_train_seqJ: raise ValueError("Joint train split failed.")
            except Exception as e: print(f"ERROR prep joint data: {e}"); return


        # Create Final DataLoaders
        print("\nCreating final DataLoaders...")
        # Standard CNN1 Loaders
        if f_train_p1: cnn1_train_loader = DataLoader(TennisFrameDataset(f_train_p1, f_train_t1, augment=True), batch_size=cnn1_trainhp_params['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        if f_val_p1: cnn1_val_loader = DataLoader(TennisFrameDataset(f_val_p1, f_val_t1, augment=False), batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        if f_test_p1: cnn1_test_loader = DataLoader(TennisFrameDataset(f_test_p1, f_test_t1, augment=False), batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        print(f"Std CNN1 Loaders: Tr={len(cnn1_train_loader or [])}, Vl={len(cnn1_val_loader or [])}, Ts={len(cnn1_test_loader or [])}")

        # Standard CNN2 Loaders (use val_test_cnn2_seq_len determined earlier)
        if f_train_seq2: cnn2_train_loader = DataLoader(BallLandingDataset(f_train_seq2, n_frames_sequence=val_test_cnn2_seq_len, augment=True), batch_size=cnn2_trainhp_params['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        if f_val_seq2: cnn2_val_loader = DataLoader(BallLandingDataset(f_val_seq2, n_frames_sequence=val_test_cnn2_seq_len, augment=False), batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        if f_test_seq2: cnn2_test_loader = DataLoader(BallLandingDataset(f_test_seq2, n_frames_sequence=val_test_cnn2_seq_len, augment=False), batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        print(f"Std CNN2 Loaders (using {val_test_cnn2_seq_len} frames): Tr={len(cnn2_train_loader or [])}, Vl={len(cnn2_val_loader or [])}, Ts={len(cnn2_test_loader or [])}")

        # Joint Training Loaders
        if args.run_joint_training and final_joint_splits:
             f_train_seqJ = final_joint_splits[0]
             joint_train_loader = DataLoader(JointPredictionDataset(f_train_seqJ, n_frames_context=config.JOINT_DATASET_CONTEXT_FRAMES, augment=True), batch_size=config.DEFAULT_JOINT_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
             # *** Use the correctly sized cnn2_val_loader for joint validation ***
             joint_val_loader = cnn2_val_loader # Reuse standard CNN2 val loader
             joint_test_loader = cnn2_test_loader # Use std test loader for consistency
             print(f"Joint Loaders: Tr={len(joint_train_loader or [])}, Vl={len(joint_val_loader or [])}, Ts={len(joint_test_loader or [])}")


    # --- 6. Final Model Training (Standard or Joint) ---
    if args.run_final_training: # Standard Individual Training
         print("\n" + "#"*20 + " Starting Standard Final Training " + "#"*20)
         if cnn1_train_loader is None or cnn1_val_loader is None: print("ERR: Std CNN1 loaders needed."); return
         # --- Train Standard CNN1 ---
         print("\n--- Final Training: Standard CNN1 ---")
         if 'filters' not in cnn1_arch_params: print("ERR: CNN1 arch filters missing."); return
         cnn1_init_params_std = {
             'block_filters': tuple(cnn1_arch_params['filters']),
             'fc_size': cnn1_arch_params['fc_size'],
             'dropout_rate': cnn1_arch_params.get('dropout', cnn1_arch_params.get('dropout_rate', config.DEFAULT_CNN1_DROPOUT))
         }
         cnn1_model_std = HitFrameRegressorFinal(**cnn1_init_params_std).to(config.DEVICE)
         criterion1 = nn.MSELoss(); optimizer1 = optim.Adam(cnn1_model_std.parameters(), lr=cnn1_trainhp_params['learning_rate'])
         model1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
         history1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_history.csv')
         train_model(model=cnn1_model_std, model_name="CNN1 Final", train_loader=cnn1_train_loader, val_loader=cnn1_val_loader, criterion=criterion1, optimizer=optimizer1, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS, early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE, min_improvement=config.DEFAULT_MIN_IMPROVEMENT, results_save_path=history1_path, best_model_save_path=model1_path)
         del cnn1_model_std, optimizer1, criterion1; gc.collect(); torch.cuda.empty_cache()

         if cnn2_train_loader is None or cnn2_val_loader is None: print("ERR: Std CNN2 loaders needed."); return
         # --- Train Standard CNN2 ---
         print("\n--- Final Training: Standard CNN2 ---")
         # Use standard seq len for input channels
         cnn2_seq_len_std_train = cnn2_dataprep_params.get('n_frames_sequence_cnn2', config.DEFAULT_N_FRAMES_SEQUENCE_CNN2)
         if cnn2_seq_len_std_train % 2 == 0: cnn2_seq_len_std_train += 1 # Ensure odd
         cnn2_input_channels_std = cnn2_seq_len_std_train * 3
         # Use loaded/default CNN2 arch params
         if 'conv_filters' not in cnn2_arch_params: print("ERR: CNN2 arch conv_filters missing."); return
         if 'fc_sizes' not in cnn2_arch_params: print("ERR: CNN2 arch fc_sizes missing."); return
         cnn2_init_params_std = {
             'input_channels': cnn2_input_channels_std,
             'conv_filters': tuple(cnn2_arch_params['conv_filters']),
             'fc_sizes': tuple(cnn2_arch_params['fc_sizes']),
             'dropout_rate': cnn2_arch_params.get('dropout', config.DEFAULT_CNN2_DROPOUT)
         }
         cnn2_model_std = LandingPointCNN(**cnn2_init_params_std).to(config.DEVICE)
         criterion2 = nn.MSELoss(); optimizer2 = optim.Adam(cnn2_model_std.parameters(), lr=cnn2_trainhp_params['learning_rate'])
         model2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_final.pth')
         history2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_final_training_history.csv')
         train_model(model=cnn2_model_std, model_name="CNN2 Final", train_loader=cnn2_train_loader, val_loader=cnn2_val_loader, criterion=criterion2, optimizer=optimizer2, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS, early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE, min_improvement=config.DEFAULT_MIN_IMPROVEMENT, results_save_path=history2_path, best_model_save_path=model2_path)
         del cnn2_model_std, optimizer2, criterion2; gc.collect(); torch.cuda.empty_cache()
         print("\n" + "#"*20 + " Standard Final Training Finished " + "#"*20)


    if args.run_joint_training:
         if joint_train_loader is None or joint_val_loader is None or config.OPTIMIZED_R1_INT is None or config.OPTIMIZED_R2_INT is None:
              print("ERROR: Cannot run joint training without joint data loaders & optimized R1/R2.")
              return
         print("\n" + "#"*20 + " Starting Joint Final Training (Validation May Be Skipped/Inaccurate) " + "#"*20)
         # Instantiate models (use best ARCH for CNN1)
         if 'filters' not in cnn1_arch_params: print("ERR: CNN1 arch filters missing."); return
         cnn1_init_params = {
             'block_filters': tuple(cnn1_arch_params['filters']),
             'fc_size': cnn1_arch_params['fc_size'],
             'dropout_rate': cnn1_arch_params.get('dropout', cnn1_arch_params.get('dropout_rate', config.DEFAULT_CNN1_DROPOUT))
         }
         cnn1_model_joint = HitFrameRegressorFinal(**cnn1_init_params).to(config.DEVICE)

         # Instantiate Joint CNN2 (use dynamic length and best CNN2 arch)
         dynamic_cnn2_seq_len = config.OPTIMIZED_R1_INT + config.OPTIMIZED_R2_INT + 1
         cnn2_input_channels_joint = dynamic_cnn2_seq_len * 3
         print(f"Instantiating Joint CNN2 with {cnn2_input_channels_joint} input channels ({dynamic_cnn2_seq_len} frames)")
         if 'conv_filters' not in cnn2_arch_params: print("ERR: CNN2 arch conv_filters missing."); return
         if 'fc_sizes' not in cnn2_arch_params: print("ERR: CNN2 arch fc_sizes missing."); return
         cnn2_init_params_joint = {
             'input_channels': cnn2_input_channels_joint,
             'conv_filters': tuple(cnn2_arch_params['conv_filters']),
             'fc_sizes': tuple(cnn2_arch_params['fc_sizes']),
             'dropout_rate': cnn2_arch_params.get('dropout', config.DEFAULT_CNN2_DROPOUT)
         }
         cnn2_model_joint = LandingPointCNN(**cnn2_init_params_joint).to(config.DEVICE)

         # Combined optimizer
         combined_params = list(cnn1_model_joint.parameters()) + list(cnn2_model_joint.parameters())
         optimizer_joint = optim.Adam(combined_params, lr=config.DEFAULT_JOINT_LR)
         # Paths
         model1_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_joint.pth')
         model2_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_joint.pth')
         history_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'joint_final_training_history.csv')

         # Set val_loader to None to disable validation within the loop for now
         # due to channel mismatch issues unless resolved.
         print("INFO: Disabling validation loop within train_joint_model due to potential channel mismatch.")
         train_joint_model(
             cnn1_model=cnn1_model_joint, cnn2_model=cnn2_model_joint, model_name="Joint Final",
             train_loader=joint_train_loader,
             val_loader=None, # Pass None to skip validation loop
             optimizer=optimizer_joint, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS,
             penalty_weight=config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT,
             R1=config.OPTIMIZED_R1_INT, R2=config.OPTIMIZED_R2_INT, # Pass optimized integers
             early_stopping_patience=0, # Disable ES if no validation
             min_improvement=config.DEFAULT_MIN_IMPROVEMENT,
             results_save_path=history_joint_path,
             best_model_save_path_cnn1=model1_joint_path, best_model_save_path_cnn2=model2_joint_path
         )
         del cnn1_model_joint, cnn2_model_joint, optimizer_joint; gc.collect(); torch.cuda.empty_cache()
         print("\n" + "#"*20 + " Joint Training Finished " + "#"*20)


    # --- 7. Evaluation ---
    if args.run_evaluation:
         print("\n" + "#"*20 + " Starting Evaluation " + "#"*20)
         # Evaluate Standard Models
         print("\n--- Evaluating Standard Models ---")
         model1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
         model2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_final.pth')

         # Eval CNN1
         if cnn1_test_loader is None: print("WARN: Std CNN1 Test loader not available for evaluation.")
         else:
             cnn1_model_eval = load_final_cnn1_model(model1_path, config.DEVICE, cnn1_arch_params)
             if cnn1_model_eval:
                 evaluate_model(cnn1_model_eval, "CNN1 Final", cnn1_test_loader, nn.MSELoss(), config.DEVICE)
                 del cnn1_model_eval; gc.collect(); torch.cuda.empty_cache()

         # Eval CNN2
         if cnn2_test_loader is None: print("WARN: Std CNN2 Test loader not available for evaluation.")
         else:
             # Instantiate standard CNN2 with correct arch for evaluation
             cnn2_seq_len_std_eval = cnn2_dataprep_params.get('n_frames_sequence_cnn2', config.DEFAULT_N_FRAMES_SEQUENCE_CNN2)
             if cnn2_seq_len_std_eval % 2 == 0: cnn2_seq_len_std_eval += 1 # Ensure odd
             cnn2_input_channels_std_eval = cnn2_seq_len_std_eval * 3
             if 'conv_filters' not in cnn2_arch_params or 'fc_sizes' not in cnn2_arch_params:
                  print("ERR: Cannot eval std CNN2 without arch params.")
             else:
                 cnn2_init_params_std_eval = {
                     'input_channels': cnn2_input_channels_std_eval,
                     'conv_filters': tuple(cnn2_arch_params['conv_filters']),
                     'fc_sizes': tuple(cnn2_arch_params['fc_sizes']),
                     'dropout_rate': cnn2_arch_params.get('dropout', config.DEFAULT_CNN2_DROPOUT)
                 }
                 if os.path.exists(model2_path):
                      cnn2_model_eval_instance = LandingPointCNN(**cnn2_init_params_std_eval).to(config.DEVICE)
                      try:
                           cnn2_model_eval_instance.load_state_dict(torch.load(model2_path, map_location=config.DEVICE))
                           print("Loaded standard CNN2 weights for evaluation.")
                           evaluate_model(cnn2_model_eval_instance, "CNN2 Final", cnn2_test_loader, nn.MSELoss(), config.DEVICE)
                           del cnn2_model_eval_instance; gc.collect(); torch.cuda.empty_cache()
                      except Exception as e: print(f"Failed to load/eval standard CNN2: {e}")
                 else: print(f"Std CNN2 model file not found: {model2_path}")


         # Evaluate Joint Models (Pipeline Eval Needed)
         print("\n--- Evaluating Joint Models (Requires Pipeline Evaluation on Test Set) ---")
         model1_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_joint.pth')
         model2_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_joint.pth')
         if os.path.exists(model1_joint_path) and os.path.exists(model2_joint_path):
              print("Joint model files exist. Run prediction pipeline on test set for full evaluation.")
              # TODO: Implement pipeline evaluation loop here if desired
         else:
              print("Joint model files not found.")

         # Plot Histories
         print("\nPlotting training histories...")
         history1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_history.csv'); plot_path1 = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_plot.png')
         if os.path.exists(history1_path): plot_training_history(history1_path, "CNN1 (Standard)", plot_path1)
         history2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_final_training_history.csv'); plot_path2 = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_final_training_plot.png')
         if os.path.exists(history2_path): plot_training_history(history2_path, "CNN2 (Standard)", plot_path2)
         history_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'joint_final_training_history.csv'); plot_path_joint = os.path.join(config.PROJECT_OUTPUT_PATH, 'joint_final_training_plot.png')
         if os.path.exists(history_joint_path): plot_training_history(history_joint_path, "Joint Training", plot_path_joint)


    # --- 8. Prediction Pipeline Example ---
    if args.run_prediction:
        print("\n" + "#"*20 + " Running Prediction Example " + "#"*20)
        # Choose which models to use
        use_joint_models_for_pred = True # Set to False to use standard models

        # Define paths and params based on choice
        if use_joint_models_for_pred:
            print("Using JOINT models for prediction example.")
            model1_pred_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_joint.pth')
            model2_pred_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_joint.pth')
            pred_r1 = config.OPTIMIZED_R1_INT
            pred_r2 = config.OPTIMIZED_R2_INT
            if pred_r1 is None or pred_r2 is None: # Load R1/R2 if needed
                  loaded_weights = load_json_params(bayes_weights_json_path, "Bayes")
                  if loaded_weights:
                      pred_r1 = int(round(loaded_weights.get('R1', 5)))
                      pred_r2 = int(round(loaded_weights.get('R2', 5)))
                  else: print("ERR: Need R1/R2 for joint prediction."); return
            pred_cnn2_seq_len = pred_r1 + pred_r2 + 1
            pred_cnn2_input_channels = pred_cnn2_seq_len * 3
        else:
            print("Using STANDARD models for prediction example.")
            model1_pred_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
            model2_pred_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_final.pth')
            pred_cnn2_seq_len = cnn2_dataprep_params.get('n_frames_sequence_cnn2', config.DEFAULT_N_FRAMES_SEQUENCE_CNN2)
            if pred_cnn2_seq_len % 2 == 0: pred_cnn2_seq_len +=1 # Ensure odd for consistency
            pred_cnn2_input_channels = pred_cnn2_seq_len * 3
            # Derive R1/R2 for the prediction function signature
            pred_r1 = pred_cnn2_seq_len // 2
            pred_r2 = pred_cnn2_seq_len // 2

        # Load models
        cnn1_pred_model = load_final_cnn1_model(model1_pred_path, config.DEVICE, cnn1_arch_params)
        # Instantiate CNN2 with correct arch before loading state_dict
        if 'conv_filters' not in cnn2_arch_params or 'fc_sizes' not in cnn2_arch_params:
             print("ERR: Cannot run prediction without CNN2 arch params.")
             cnn2_pred_model = None
        else:
             cnn2_init_params_pred = {
                 'input_channels': pred_cnn2_input_channels,
                 'conv_filters': tuple(cnn2_arch_params['conv_filters']),
                 'fc_sizes': tuple(cnn2_arch_params['fc_sizes']),
                 'dropout_rate': cnn2_arch_params.get('dropout', config.DEFAULT_CNN2_DROPOUT)
             }
             cnn2_pred_model_instance = LandingPointCNN(**cnn2_init_params_pred).to(config.DEVICE)
             cnn2_pred_model = None # Reset
             if os.path.exists(model2_pred_path):
                 try:
                      cnn2_pred_model_instance.load_state_dict(torch.load(model2_pred_path, map_location=config.DEVICE))
                      cnn2_pred_model_instance.eval()
                      print(f"Loaded prediction CNN2 weights from {os.path.basename(model2_pred_path)}")
                      cnn2_pred_model = cnn2_pred_model_instance # Use the loaded instance
                 except Exception as e: print(f"ERROR loading prediction CNN2 weights: {e}")
             else: print(f"Prediction CNN2 model file not found: {model2_pred_path}")


        if not cnn1_pred_model or not cnn2_pred_model:
            print("Error: One or both models not loaded for prediction.")
        else:
            # Find example directory
            example_dir = None
            actual_coords_example = None
            try: # Use standard test set for finding example
                if final_cnn2_splits: # Check if splits were created
                     test_sequences_std = final_cnn2_splits[2]
                     if test_sequences_std:
                          example_sequence_info = random.choice(test_sequences_std)
                          if example_sequence_info.get('sequence_paths'):
                               example_dir = os.path.dirname(example_sequence_info['sequence_paths'][0])
                               actual_coords_example = example_sequence_info.get('target_coords')
                               print(f"Selected example directory: {os.path.basename(example_dir)}")
                               if actual_coords_example: print(f"Actual Landing Coords: ({actual_coords_example[0]:.3f}, {actual_coords_example[1]:.3f})")
                          else: print("WARN: Example sequence dict missing 'sequence_paths'.")
                     else: print("WARN: Standard Test sequence list empty.")
                else: print("WARN: final_cnn2_splits not defined, cannot select test example.")
            except Exception as e: print(f"WARN: Error selecting example from test splits: {e}")

            # Fallback
            if not example_dir and dataset_base_path:
                 frames_root = os.path.join(dataset_base_path, 'Frames')
                 try:
                      potential_dirs = [os.path.join(dp, d) for dp, dn, fn in os.walk(frames_root) for d in dn if re.match(r'^(ICT|IST|OCT|OST)\d+$', d)]
                      if potential_dirs:
                           example_dir = random.choice(potential_dirs)
                           print(f"Using fallback example directory: {os.path.basename(example_dir)}")
                 except Exception as e: print(f"Error finding fallback directory: {e}")

            if example_dir and os.path.isdir(example_dir):
                predicted_coords, predicted_hit_path = predict_hit_and_landing(
                    cnn1_model=cnn1_pred_model,
                    cnn2_model=cnn2_pred_model,
                    frames_directory=example_dir,
                    R1=pred_r1, # Pass R1 (can be None for standard)
                    R2=pred_r2, # Pass R2 (can be None for standard)
                    device=config.DEVICE
                )

                # Display results
                if predicted_coords and predicted_hit_path:
                    print("\n--- Example Pipeline Results ---")
                    print(f"Directory: {os.path.basename(example_dir)}")
                    print(f"Predicted Hit Frame: {os.path.basename(predicted_hit_path)}")
                    print(f"Predicted Landing (Norm): ({predicted_coords[0]:.4f}, {predicted_coords[1]:.4f})")

                    dl, dr, db = denormalize_coordinates(predicted_coords[0], predicted_coords[1])
                    if dl is not None: print(f"Predicted Landing (Approx. Meters): L={dl:.2f}m, R={dr:.2f}m, Baseline={db:.2f}m")

                    if actual_coords_example:
                        err_x = abs(predicted_coords[0] - actual_coords_example[0])
                        err_y = abs(predicted_coords[1] - actual_coords_example[1])
                        euc_dist_norm = np.sqrt(err_x**2 + err_y**2)
                        print(f"Normalized Euclidean Error vs Actual: {euc_dist_norm:.4f}")

                    # Visualize
                    try:
                        img_hit = cv2.imread(predicted_hit_path)
                        if img_hit is not None:
                             img_rgb = cv2.cvtColor(img_hit, cv2.COLOR_BGR2RGB)
                             h, w, _ = img_rgb.shape
                             marker_x = int(predicted_coords[0] * w)
                             marker_y = int((1.0 - predicted_coords[1]) * h) # Y=0 baseline -> Image Y=0 top

                             plt.figure(figsize=(7, 6))
                             plt.imshow(img_rgb)
                             plt.plot(marker_x, marker_y, 'ro', markersize=10, alpha=0.8, label=f'Pred ({predicted_coords[0]:.2f},{predicted_coords[1]:.2f})')

                             if actual_coords_example:
                                 actual_marker_x = int(actual_coords_example[0] * w)
                                 actual_marker_y = int((1.0 - actual_coords_example[1]) * h)
                                 plt.plot(actual_marker_x, actual_marker_y, 'go', markersize=10, alpha=0.6, label=f'Actual ({actual_coords_example[0]:.2f},{actual_coords_example[1]:.2f})')

                             plt.title(f"Prediction Example: {os.path.basename(example_dir)}")
                             plt.legend()
                             plt.axis('off')
                             plt.tight_layout()
                             plot_save_path = os.path.join(config.PROJECT_OUTPUT_PATH, f'prediction_example_{os.path.basename(example_dir)}.png')
                             plt.savefig(plot_save_path)
                             print(f"Saved prediction visualization to: {plot_save_path}")
                             plt.show()
                             plt.close() # Close the plot explicitly
                    except Exception as e: print(f"Error visualizing prediction: {e}")
                else:
                     print("Pipeline prediction failed for example.")

            else: print("Could not find valid example dir for prediction.")
            del cnn1_pred_model, cnn2_pred_model; gc.collect(); torch.cuda.empty_cache()


    end_time_main = time.time()
    print(f"\nTotal execution time: {end_time_main - start_time_main:.2f} seconds.")
    print("="*50)
    print(" Pipeline Finished ".center(50, "="))
    print("="*50)


# --- Entry Point ---
if __name__ == "__main__":
    # Needed for Windows multiprocessing in DataLoader with num_workers > 0
    if os.name == 'nt':
        try:
            # Check if context is already set before trying to set it
            if torch.multiprocessing.get_start_method(allow_none=True) is None:
                torch.multiprocessing.set_start_method('spawn')
                # print("Set multiprocessing start method to 'spawn'.") # Debug
        except RuntimeError as e:
            # Ignore error if context has already been set by another module/library
            if "context has already been set" not in str(e):
                print(f"Warning: Could not set multiprocessing start method: {e}")
        except Exception as e: # Catch other potential errors
            print(f"Warning: Error setting multiprocessing start method: {e}")


    parser = argparse.ArgumentParser(description="ST311 Tennis Analysis Pipeline")
    parser.add_argument('--download_data', action='store_true', help='Attempt to download dataset from Kaggle.')
    parser.add_argument('--run_grid_search', action='store_true', help='Run grid search for Arch/HP.')
    parser.add_argument('--run_bayesian_opt', action='store_true', help='Run Bayesian optimization for CNN1 h(x) weighting.')
    parser.add_argument('--run_final_training', action='store_true', help='Run standard final training using best/default parameters.')
    parser.add_argument('--run_joint_training', action='store_true', help='Run joint end-to-end training using optimized parameters.')
    parser.add_argument('--run_evaluation', action='store_true', help='Evaluate final models on the test set.')
    parser.add_argument('--run_prediction', action='store_true', help='Run the prediction pipeline on an example.')
    parser.add_argument('--all', action='store_true', help='Run all steps (download, grid search, bayes opt, joint training, evaluation, prediction).')

    args = parser.parse_args()

    if args.all:
        args.download_data = True
        args.run_grid_search = True
        args.run_bayesian_opt = True
        # args.run_final_training = True # Optionally run standard training as well
        args.run_joint_training = True
        args.run_evaluation = True
        args.run_prediction = True

    # Check if at least one action is requested
    if not any([args.download_data, args.run_grid_search, args.run_bayesian_opt,
                args.run_final_training, args.run_joint_training,
                args.run_evaluation, args.run_prediction]):
        print("No action specified. Use --help to see options.")
        print("Example usage:")
        print("  python main.py --all")
        print("  python main.py --run_bayesian_opt --run_joint_training --run_evaluation")
        print("  python main.py --run_prediction")
    else:
        # Print config details once
        print("="*50)
        print(" ST311 Tennis Analysis Pipeline - Initializing ".center(50, "="))
        print(f"Output Directory: {config.PROJECT_OUTPUT_PATH}")
        print(f"Using Device: {config.DEVICE}")
        print(f"Arguments: {args}")
        print("="*50)
        main(args)