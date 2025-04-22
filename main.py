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
import re

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
from datasets import TennisFrameDataset, BallLandingDataset, JointPredictionDataset # Add Joint
from models import HitFrameRegressorFinal, LandingPointCNN
from training import train_model, evaluate_model, train_joint_model # Add Joint
from grid_search import (run_cnn1_arch_search, run_cnn1_trainhp_search, # Removed CNN1 DataPrep grid search
                         run_cnn2_dataprep_search, run_cnn2_trainhp_search)
from bayesian_optimizer import run_bayesian_optimization # NEW Import
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
    cnn2_dataprep_params = {'n_frames_sequence_cnn2': config.DEFAULT_N_FRAMES_SEQUENCE_CNN2} # For standard CNN2
    cnn2_trainhp_params = {'learning_rate': config.DEFAULT_CNN2_LR, 'batch_size': config.DEFAULT_CNN2_BATCH_SIZE}


    # --- 4. Optional Searches (Grid for Arch/HP, BayesOpt for Weighting) ---
    final_cnn1_splits = None # Uses linear weights if only standard training
    final_cnn2_splits = None # Standard CNN2 splits
    final_joint_splits = None # Splits for JointPredictionDataset

    if args.run_grid_search or args.run_bayesian_opt:
        if df_full is None:
            print("ERROR: Cannot run search/optimization without loaded metadata (df_full).")
            return

        print("\n" + "#"*20 + " Starting Hyperparameter Optimization " + "#"*20)

        # --- CNN1 Architecture Search (Grid Search) ---
        if args.run_grid_search:
             print("\nUsing LINEAR weighting for CNN1 Arch search prep...")
             initial_df_processed = apply_linear_weighting_to_df(
                 df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY
             )
             initial_cnn1_splits = balance_and_split_data(initial_df_processed, config.DEFAULT_BALANCE_RATIO)
             best_filters, best_fc, best_dropout = run_cnn1_arch_search(df_full, initial_cnn1_splits, config.DEVICE)
             cnn1_arch_params = {'filters': best_filters, 'fc_size': best_fc, 'dropout': best_dropout}
        else:
             loaded_arch = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_architecture.json'), "CNN1 Arch")
             if loaded_arch: cnn1_arch_params = loaded_arch

        # --- CNN1 h(x) Weighting Parameter Search (Bayesian Optimization) ---
        if args.run_bayesian_opt:
             # Optimize h(x) params based on individual CNN1 training against h(x) targets
             cnn1_weighting_params = run_bayesian_optimization(df_full, tuple(cnn1_arch_params.values()), config.DEVICE)
             # R1/R2 integers are now stored in config.OPTIMIZED_R1_INT/R2_INT by the function
        else:
             # Load if not running opt
             loaded_weights = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_bayesian_weights.json'), "CNN1 Bayesian Weights")
             if loaded_weights:
                  cnn1_weighting_params = loaded_weights
                  # Set global R1/R2 from loaded file
                  config.OPTIMIZED_R1_INT = int(round(loaded_weights.get('R1', 5))) # Default if missing
                  config.OPTIMIZED_R2_INT = int(round(loaded_weights.get('R2', 5)))
                  print(f"Loaded Optimized R1 (int): {config.OPTIMIZED_R1_INT}, R2 (int): {config.OPTIMIZED_R2_INT}")
             else:
                  print("Warning: Bayesian weight params file not found. Cannot run Joint Training without them.")
                  if args.run_joint_training: return # Exit if joint training requested

        # --- CNN1 Training HP Search (Grid Search) ---
        # Use LINEAR weighting for this standard CNN1 HP search
        if args.run_grid_search:
             print("\nRegenerating CNN1 splits using LINEAR weights for HP Search...")
             df_processed_linear = apply_linear_weighting_to_df(
                 df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY
             )
             splits_for_hp_search = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
             best_lr_cnn1, best_bs_cnn1 = run_cnn1_trainhp_search(splits_for_hp_search, tuple(cnn1_arch_params.values()), config.DEVICE)
             cnn1_trainhp_params = {'learning_rate': best_lr_cnn1, 'batch_size': best_bs_cnn1}
        else:
             loaded_hp1 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_training_hp.json'), "CNN1 TrainHP")
             if loaded_hp1: cnn1_trainhp_params = loaded_hp1

        # --- CNN2 Searches (Sequence Length & HPs for standard CNN2) ---
        if args.run_grid_search:
             try: landing_df_indexed = load_landing_data(dataset_base_path)
             except Exception as e: print(f"ERROR loading landing data: {e}"); return

             # Need a balanced DF for CNN2 search. Use linear one used for HP search.
             if 'df_processed_linear' not in locals(): # Ensure it exists
                  df_processed_linear = apply_linear_weighting_to_df(df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY)
             if 'splits_for_hp_search' not in locals():
                  splits_for_hp_search = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
             cnn1_balanced_df_for_cnn2_search = splits_for_hp_search[0]

             # CNN2 Seq Len Search (standard CNN2)
             best_seq_len, final_cnn2_splits_unpacked = run_cnn2_dataprep_search(cnn1_balanced_df_for_cnn2_search, landing_df_indexed, config.DEVICE)
             cnn2_dataprep_params = {'n_frames_sequence_cnn2': best_seq_len}
             final_cnn2_splits = (final_cnn2_splits_unpacked[0], final_cnn2_splits_unpacked[1], final_cnn2_splits_unpacked[2])

             # CNN2 Training HP Search (standard CNN2)
             best_lr_cnn2, best_bs_cnn2 = run_cnn2_trainhp_search(final_cnn2_splits, best_seq_len, config.DEVICE)
             cnn2_trainhp_params = {'learning_rate': best_lr_cnn2, 'batch_size': best_bs_cnn2}
        else:
             loaded_dp2 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_dataprep.json'), "CNN2 DataPrep")
             if loaded_dp2: cnn2_dataprep_params = loaded_dp2
             loaded_hp2 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_training_hp.json'), "CNN2 TrainHP")
             if loaded_hp2: cnn2_trainhp_params = loaded_hp2

        print("\n" + "#"*20 + " Hyperparameter Optimization Finished " + "#"*20)

    else: # Neither grid search nor bayes opt requested, load all from files
        print("\nSkipping optimization. Loading best parameters from JSON files...")
        loaded_arch = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_architecture.json'), "CNN1 Arch")
        if loaded_arch: cnn1_arch_params = loaded_arch
        loaded_weights = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_bayesian_weights.json'), "CNN1 Bayesian Weights")
        if loaded_weights:
             cnn1_weighting_params = loaded_weights
             config.OPTIMIZED_R1_INT = int(round(loaded_weights.get('R1', 5)))
             config.OPTIMIZED_R2_INT = int(round(loaded_weights.get('R2', 5)))
             print(f"Loaded Optimized R1 (int): {config.OPTIMIZED_R1_INT}, R2 (int): {config.OPTIMIZED_R2_INT}")
        else:
            print("Warning: Bayesian weight params file not found. Cannot run Joint Training.")
            if args.run_joint_training: return

        loaded_hp1 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_training_hp.json'), "CNN1 TrainHP")
        if loaded_hp1: cnn1_trainhp_params = loaded_hp1
        loaded_dp2 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_dataprep.json'), "CNN2 DataPrep")
        if loaded_dp2: cnn2_dataprep_params = loaded_dp2
        loaded_hp2 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_training_hp.json'), "CNN2 TrainHP")
        if loaded_hp2: cnn2_trainhp_params = loaded_hp2


    # --- 5. Final Data Preparation (Standard & Joint) ---
    # Declare loaders here to handle scope if only eval/pred is run
    cnn1_train_loader, cnn1_val_loader, cnn1_test_loader = None, None, None
    cnn2_train_loader, cnn2_val_loader, cnn2_test_loader = None, None, None
    joint_train_loader, joint_val_loader, joint_test_loader = None, None, None

    if args.run_final_training or args.run_joint_training or args.run_evaluation:
        if df_full is None:
             print("ERROR: df_full needed for data preparation.")
             return
        if args.run_joint_training and cnn1_weighting_params is None:
             print("ERROR: Bayesian weighting params needed for joint training data prep.")
             return

        print("\nPreparing final datasets...")
        # Regenerate CNN1 data splits using LINEAR weights for standard training/eval
        df_processed_linear = apply_linear_weighting_to_df(
            df_full, config.DEFAULT_LINEAR_N_FRAMES_WEIGHTING, config.DEFAULT_LINEAR_WEIGHT_DECAY
        )
        final_cnn1_splits = balance_and_split_data(df_processed_linear, config.DEFAULT_BALANCE_RATIO)
        cnn1_balanced_df_linear = final_cnn1_splits[0] # Store balanced df from linear weights
        _, f_train_p1, f_train_t1, f_val_p1, f_val_t1, f_test_p1, f_test_t1 = final_cnn1_splits
        if not f_train_p1 or not f_val_p1 or not f_test_p1: print("ERROR: Final std CNN1 split failed."); return

        # Prepare standard CNN2 sequences/splits
        try:
            landing_df_indexed = load_landing_data(dataset_base_path)
            final_sequences_cnn2 = get_sequences_for_cnn2(
                cnn1_balanced_df_linear, landing_df_indexed, cnn2_dataprep_params['n_frames_sequence_cnn2']
            )
            if not final_sequences_cnn2: raise ValueError("No std CNN2 sequences.")
            f_train_seq2, f_val_seq2, f_test_seq2 = split_sequences(final_sequences_cnn2)
            final_cnn2_splits = (f_train_seq2, f_val_seq2, f_test_seq2)
            if not f_train_seq2 or not f_val_seq2 or not f_test_seq2: raise ValueError("Std CNN2 split failed.")
        except Exception as e: print(f"ERROR prep standard CNN2: {e}"); return

        # Prepare JOINT sequences/splits using BEST Bayesian h(x) weights (only if needed)
        if args.run_joint_training:
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
                f_train_seqJ, f_val_seqJ, f_test_seqJ = split_sequences(final_sequences_joint)
                # Use standard CNN2 val/test set for joint validation/testing for consistency
                final_joint_splits = (f_train_seqJ, f_val_seq2, f_test_seq2) # Train J, Val/Test Std
                if not f_train_seqJ: raise ValueError("Joint train split failed.")
            except Exception as e: print(f"ERROR prep joint data: {e}"); return

        # Create Final DataLoaders
        print("\nCreating final DataLoaders...")
        # Standard CNN1 Loaders (Use final_cnn1_splits based on linear weights)
        if f_train_p1: cnn1_train_loader = DataLoader(TennisFrameDataset(f_train_p1, f_train_t1, augment=True), batch_size=cnn1_trainhp_params['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        if f_val_p1: cnn1_val_loader = DataLoader(TennisFrameDataset(f_val_p1, f_val_t1, augment=False), batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        if f_test_p1: cnn1_test_loader = DataLoader(TennisFrameDataset(f_test_p1, f_test_t1, augment=False), batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        print(f"Std CNN1 Loaders: Tr={len(cnn1_train_loader or [])}, Vl={len(cnn1_val_loader or [])}, Ts={len(cnn1_test_loader or [])}")

        # Standard CNN2 Loaders (Use final_cnn2_splits)
        cnn2_seq_len_std = cnn2_dataprep_params['n_frames_sequence_cnn2']
        if f_train_seq2: cnn2_train_loader = DataLoader(BallLandingDataset(f_train_seq2, n_frames_sequence=cnn2_seq_len_std, augment=True), batch_size=cnn2_trainhp_params['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        if f_val_seq2: cnn2_val_loader = DataLoader(BallLandingDataset(f_val_seq2, n_frames_sequence=cnn2_seq_len_std, augment=False), batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        if f_test_seq2: cnn2_test_loader = DataLoader(BallLandingDataset(f_test_seq2, n_frames_sequence=cnn2_seq_len_std, augment=False), batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        print(f"Std CNN2 Loaders: Tr={len(cnn2_train_loader or [])}, Vl={len(cnn2_val_loader or [])}, Ts={len(cnn2_test_loader or [])}")

        # Joint Training Loaders (Use final_joint_splits)
        if args.run_joint_training and final_joint_splits:
             f_train_seqJ = final_joint_splits[0]
             joint_train_loader = DataLoader(JointPredictionDataset(f_train_seqJ, n_frames_context=config.JOINT_DATASET_CONTEXT_FRAMES, augment=True), batch_size=config.DEFAULT_JOINT_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
             joint_val_loader = cnn2_val_loader # Reuse standard CNN2 val loader
             joint_test_loader = cnn2_test_loader # Reuse standard CNN2 test loader for eval
             print(f"Joint Loaders: Tr={len(joint_train_loader or [])}, Vl={len(joint_val_loader or [])}, Ts={len(joint_test_loader or [])}")


    # --- 6. Final Model Training (Standard or Joint) ---
    if args.run_final_training: # Standard Individual Training
         print("\n" + "#"*20 + " Starting Standard Final Training " + "#"*20)
         if cnn1_train_loader is None or cnn1_val_loader is None: print("ERR: Std CNN1 loaders needed."); return
         # --- Train Standard CNN1 (using linear weights) ---
         print("\n--- Final Training: Standard CNN1 ---")
         cnn1_model_std = HitFrameRegressorFinal(**cnn1_arch_params).to(config.DEVICE)
         criterion1 = nn.MSELoss(); optimizer1 = optim.Adam(cnn1_model_std.parameters(), lr=cnn1_trainhp_params['learning_rate'])
         model1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
         history1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_history.csv')
         train_model(model=cnn1_model_std, model_name="CNN1 Final", train_loader=cnn1_train_loader, val_loader=cnn1_val_loader, criterion=criterion1, optimizer=optimizer1, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS, early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE, min_improvement=config.DEFAULT_MIN_IMPROVEMENT, results_save_path=history1_path, best_model_save_path=model1_path)
         del cnn1_model_std, optimizer1, criterion1; gc.collect(); torch.cuda.empty_cache()

         if cnn2_train_loader is None or cnn2_val_loader is None: print("ERR: Std CNN2 loaders needed."); return
         # --- Train Standard CNN2 ---
         print("\n--- Final Training: Standard CNN2 ---")
         cnn2_input_channels_std = cnn2_dataprep_params['n_frames_sequence_cnn2'] * 3
         cnn2_model_std = LandingPointCNN(input_channels=cnn2_input_channels_std).to(config.DEVICE)
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
         print("\n" + "#"*20 + " Starting Joint Final Training " + "#"*20)
         # Instantiate models (use best ARCH for CNN1)
         cnn1_model_joint = HitFrameRegressorFinal(**cnn1_arch_params).to(config.DEVICE)
         # Determine CNN2 input channels based on OPTIMIZED R1+R2+1
         dynamic_cnn2_seq_len = config.OPTIMIZED_R1_INT + config.OPTIMIZED_R2_INT + 1
         cnn2_input_channels_joint = dynamic_cnn2_seq_len * 3
         print(f"Instantiating Joint CNN2 with {cnn2_input_channels_joint} input channels ({dynamic_cnn2_seq_len} frames)")
         cnn2_model_joint = LandingPointCNN(input_channels=cnn2_input_channels_joint).to(config.DEVICE)
         # Combined optimizer
         combined_params = list(cnn1_model_joint.parameters()) + list(cnn2_model_joint.parameters())
         optimizer_joint = optim.Adam(combined_params, lr=config.DEFAULT_JOINT_LR)
         # Paths
         model1_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_joint.pth')
         model2_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_joint.pth')
         history_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'joint_final_training_history.csv')

         train_joint_model(
             cnn1_model=cnn1_model_joint, cnn2_model=cnn2_model_joint, model_name="Joint Final",
             train_loader=joint_train_loader, val_loader=joint_val_loader, # Val uses std CNN2 loader
             optimizer=optimizer_joint, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS,
             penalty_weight=config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT,
             R1=config.OPTIMIZED_R1_INT, R2=config.OPTIMIZED_R2_INT, # Pass optimized integers
             early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE,
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

         if cnn1_test_loader is None: print("WARN: Std CNN1 Test loader not available for evaluation.")
         else:
             cnn1_model_eval = load_final_cnn1_model(model1_path, config.DEVICE, cnn1_arch_params)
             if cnn1_model_eval:
                 evaluate_model(cnn1_model_eval, "CNN1 Final", cnn1_test_loader, nn.MSELoss(), config.DEVICE)
                 del cnn1_model_eval; gc.collect(); torch.cuda.empty_cache()

         if cnn2_test_loader is None: print("WARN: Std CNN2 Test loader not available for evaluation.")
         else:
             cnn2_input_channels_std = cnn2_dataprep_params['n_frames_sequence_cnn2'] * 3
             cnn2_model_eval = load_final_cnn2_model(model2_path, config.DEVICE, cnn2_input_channels_std)
             if cnn2_model_eval:
                 evaluate_model(cnn2_model_eval, "CNN2 Final", cnn2_test_loader, nn.MSELoss(), config.DEVICE)
                 del cnn2_model_eval; gc.collect(); torch.cuda.empty_cache()

         # Evaluate Joint Models (Pipeline Eval Needed)
         print("\n--- Evaluating Joint Models (Requires Pipeline Evaluation) ---")
         model1_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_joint.pth')
         model2_joint_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_joint.pth')
         if os.path.exists(model1_joint_path) and os.path.exists(model2_joint_path):
              print("Joint model files exist. Run prediction pipeline on test set for full evaluation.")
              # TODO: Implement pipeline evaluation loop here if desired by iterating through test set
              #       using predict_hit_and_landing and comparing with ground truth.
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
        # Choose which models to use for prediction
        use_joint_models_for_pred = True # Set to False to use standard models
        if use_joint_models_for_pred:
            print("Using JOINT models for prediction example.")
            model1_pred_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_joint.pth')
            model2_pred_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_joint.pth')
            # Need R1/R2 for the prediction pipeline
            pred_r1 = config.OPTIMIZED_R1_INT
            pred_r2 = config.OPTIMIZED_R2_INT
            if pred_r1 is None or pred_r2 is None: # Load from file if not run in this session
                  loaded_weights = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_bayesian_weights.json'), "Bayes")
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
            pred_r1 = None # Not needed directly, but derived sequence length is
            pred_r2 = None
            pred_cnn2_seq_len = cnn2_dataprep_params['n_frames_sequence_cnn2']
            pred_cnn2_input_channels = pred_cnn2_seq_len * 3
            # We still need R1/R2 for the predict_hit_and_landing function signature
            # Use dummy values or derive them from pred_cnn2_seq_len?
            # Let's derive approximate R1/R2 for the standard case
            pred_r1 = pred_cnn2_seq_len // 2
            pred_r2 = pred_cnn2_seq_len // 2


        # Load models
        cnn1_pred_model = load_final_cnn1_model(model1_pred_path, config.DEVICE, cnn1_arch_params)
        # Check if the required CNN2 input channels match the loaded model structure
        try:
             cnn2_pred_model = load_final_cnn2_model(model2_pred_path, config.DEVICE, pred_cnn2_input_channels)
        except RuntimeError as e:
            print(f"ERROR loading CNN2 model: {e}")
            print(f"Model file might have been trained with different input channels ({pred_cnn2_input_channels} required based on R1/R2 or std seq len).")
            cnn2_pred_model = None
        except Exception as e:
            print(f"ERROR loading CNN2 model: {e}")
            cnn2_pred_model = None


        if not cnn1_pred_model or not cnn2_pred_model:
            print("Error: Models not loaded for prediction.")
        else:
            # Find example directory
            example_dir = None
            actual_coords_example = None
            try: # Use standard test set for finding example
                if 'final_cnn2_splits' in locals() and final_cnn2_splits:
                     test_sequences_std = final_cnn2_splits[2]
                     if test_sequences_std:
                          example_sequence_info = random.choice(test_sequences_std)
                          example_dir = os.path.dirname(example_sequence_info['sequence_paths'][0])
                          actual_coords_example = example_sequence_info['target_coords']
                          print(f"Selected example directory: {os.path.basename(example_dir)}")
                          if actual_coords_example: print(f"Actual Landing Coords: ({actual_coords_example[0]:.3f}, {actual_coords_example[1]:.3f})")
                     else: print("WARN: Standard Test sequence list empty.")
            except NameError: print("WARN: final_cnn2_splits not defined, cannot select test example.")

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
                    R1=pred_r1, # Pass R1
                    R2=pred_r2, # Pass R2
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
            torch.multiprocessing.set_start_method('spawn', force=True)
            # print("Set multiprocessing start method to 'spawn'.") # Debug
        except RuntimeError as e:
            if "context has already been set" not in str(e):
                print(f"Warning: Could not set multiprocessing start method: {e}")


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
        # args.run_final_training = True # Don't run standard if running joint? Or run both? Let's run joint only with --all.
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