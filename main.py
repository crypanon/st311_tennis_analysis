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
import re

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.utils.data')
warnings.filterwarnings('ignore', category=FutureWarning) # Often from pandas/numpy via dependencies

# --- Project Imports ---
import config # Load configuration first
from utils import load_json_params, plot_training_history, setup_kaggle_downloader
from data_utils import (load_metadata, apply_weighting_to_df, balance_and_split_data,
                        load_landing_data, get_sequences_for_cnn2, split_sequences)
from datasets import TennisFrameDataset, BallLandingDataset
from models import HitFrameRegressorFinal, LandingPointCNN
from training import train_model, evaluate_model
from grid_search import (run_cnn1_arch_search, run_cnn1_dataprep_search, run_cnn1_trainhp_search,
                         run_cnn2_dataprep_search, run_cnn2_trainhp_search)
from prediction import (load_final_cnn1_model, load_final_cnn2_model,
                        predict_hit_and_landing, denormalize_coordinates)

# Set seed for reproducibility (optional)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # Ensure deterministic algorithms are used if needed (can impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False # Turn off benchmark if deterministic

def main(args):
    """Main execution function."""
    start_time_main = time.time()

    print("="*50)
    print(" ST311 Tennis Analysis Pipeline")
    print("="*50)
    print(f"Using Device: {config.DEVICE}")
    print(f"Output Directory: {config.PROJECT_OUTPUT_PATH}")
    print(f"Arguments: {args}")

    # --- 1. Dataset Setup ---
    dataset_base_path = config.DATASET_BASE_PATH # Get initial value (None)
    if not dataset_base_path or not os.path.exists(dataset_base_path):
        if args.download_data:
            print("\nAttempting to download dataset via Kaggle Hub...")
            dataset_base_path = setup_kaggle_downloader()
            if dataset_base_path:
                 print(f"Using downloaded dataset path: {dataset_base_path}")
            else:
                 print("Automatic download failed. Please set DATASET_BASE_PATH in config.py manually.")
                 return # Exit if data is required and download failed
        else:
            print("\nDataset path not set or invalid in config.py.")
            print("Please set DATASET_BASE_PATH or run with --download_data.")
            # Optionally proceed if data path isn't strictly needed (e.g., only loading models)
            if args.run_grid_search or args.run_final_training:
                 print("Exiting as data is required for search/training.")
                 return
            # Allow proceeding if only prediction is requested and models exist
            elif args.run_prediction:
                 print("Proceeding without dataset path (assuming models exist for prediction).")
                 dataset_base_path = None # Ensure it remains None
            else:
                 return # Exit otherwise


    # --- 2. Load Metadata (if path is valid) ---
    df_full = None
    if dataset_base_path and os.path.exists(dataset_base_path):
        csv_path = os.path.join(dataset_base_path, 'Frames/hit_frames.csv')
        try:
            df_full = load_metadata(csv_path, dataset_base_path)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            if args.run_grid_search or args.run_final_training: return
        except Exception as e:
            print(f"An unexpected error occurred during metadata loading: {e}")
            if args.run_grid_search or args.run_final_training: return
    elif args.run_grid_search or args.run_final_training:
         print("Error: Cannot proceed with search/training without a valid dataset path.")
         return


    # --- 3. Initialize Hyperparameters ---
    # Start with defaults, load best if files exist and not running grid search
    cnn1_arch_params = {'filters': config.DEFAULT_CNN1_FILTERS, 'fc_size': config.DEFAULT_CNN1_FC_SIZE, 'dropout': config.DEFAULT_CNN1_DROPOUT}
    cnn1_dataprep_params = {'n_frames_weighting': config.DEFAULT_N_FRAMES_WEIGHTING, 'weight_decay': config.DEFAULT_WEIGHT_DECAY, 'balance_ratio': config.DEFAULT_BALANCE_RATIO}
    cnn1_trainhp_params = {'learning_rate': config.DEFAULT_CNN1_LR, 'batch_size': config.DEFAULT_CNN1_BATCH_SIZE}
    cnn2_dataprep_params = {'n_frames_sequence_cnn2': config.DEFAULT_N_FRAMES_SEQUENCE_CNN2}
    cnn2_trainhp_params = {'learning_rate': config.DEFAULT_CNN2_LR, 'batch_size': config.DEFAULT_CNN2_BATCH_SIZE}

    # --- 4. Grid Search (Optional) ---
    # Data splits needed for grid search
    initial_cnn1_splits = None
    final_cnn1_splits = None
    final_cnn2_splits = None
    cnn1_balanced_df_final = None # Store the balanced df corresponding to best CNN1 data prep

    if args.run_grid_search:
        if df_full is None:
            print("ERROR: Cannot run grid search without loaded metadata (df_full).")
            return

        print("\n" + "#"*20 + " Starting Grid Search " + "#"*20)
        # Need initial weighting/split for arch search
        print("\nPerforming initial weighting and split for arch search...")
        initial_df_processed = apply_weighting_to_df(df_full, cnn1_dataprep_params['n_frames_weighting'], cnn1_dataprep_params['weight_decay'])
        initial_cnn1_splits = balance_and_split_data(initial_df_processed, cnn1_dataprep_params['balance_ratio'])

        # a) CNN1 Architecture Search
        best_filters, best_fc, best_dropout = run_cnn1_arch_search(df_full, initial_cnn1_splits, config.DEVICE)
        cnn1_arch_params = {'filters': best_filters, 'fc_size': best_fc, 'dropout': best_dropout}

        # b) CNN1 Data Prep Search
        best_n_frames, best_decay, best_balance, final_cnn1_splits = run_cnn1_dataprep_search(df_full, tuple(cnn1_arch_params.values()), config.DEVICE)
        cnn1_dataprep_params = {'n_frames_weighting': best_n_frames, 'weight_decay': best_decay, 'balance_ratio': best_balance}
        cnn1_balanced_df_final = final_cnn1_splits[0] # Get the balanced DF from the split results

        # c) CNN1 Training HP Search
        best_lr_cnn1, best_bs_cnn1 = run_cnn1_trainhp_search(final_cnn1_splits, tuple(cnn1_arch_params.values()), config.DEVICE)
        cnn1_trainhp_params = {'learning_rate': best_lr_cnn1, 'batch_size': best_bs_cnn1}

        # Load landing data needed for CNN2 search
        try:
             landing_df_indexed = load_landing_data(dataset_base_path)
        except Exception as e:
             print(f"ERROR loading landing data, cannot proceed with CNN2 search: {e}")
             return

        # d) CNN2 Data Prep (Sequence Length) Search
        best_seq_len, final_cnn2_splits_unpacked = run_cnn2_dataprep_search(cnn1_balanced_df_final, landing_df_indexed, config.DEVICE)
        cnn2_dataprep_params = {'n_frames_sequence_cnn2': best_seq_len}
        final_cnn2_splits = (final_cnn2_splits_unpacked[0], final_cnn2_splits_unpacked[1], final_cnn2_splits_unpacked[2]) # Re-pack as tuple of lists

        # e) CNN2 Training HP Search
        best_lr_cnn2, best_bs_cnn2 = run_cnn2_trainhp_search(final_cnn2_splits, best_seq_len, config.DEVICE)
        cnn2_trainhp_params = {'learning_rate': best_lr_cnn2, 'batch_size': best_bs_cnn2}

        print("\n" + "#"*20 + " Grid Search Finished " + "#"*20)

    else:
        print("\nSkipping grid search. Loading best parameters from JSON files (if they exist)...")
        # Load best parameters found previously
        loaded_arch = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_architecture.json'), "CNN1 Arch")
        if loaded_arch: cnn1_arch_params = loaded_arch

        loaded_dp1 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_dataprep.json'), "CNN1 DataPrep")
        if loaded_dp1: cnn1_dataprep_params = loaded_dp1

        loaded_hp1 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_training_hp.json'), "CNN1 TrainHP")
        if loaded_hp1: cnn1_trainhp_params = loaded_hp1

        loaded_dp2 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_dataprep.json'), "CNN2 DataPrep")
        if loaded_dp2: cnn2_dataprep_params = loaded_dp2

        loaded_hp2 = load_json_params(os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_training_hp.json'), "CNN2 TrainHP")
        if loaded_hp2: cnn2_trainhp_params = loaded_hp2

    # --- 5. Final Data Preparation ---
    # Regenerate data using the *final* determined parameters
    if args.run_final_training or args.run_evaluation: # Only needed if training or evaluating
        if df_full is None:
            print("ERROR: Cannot prepare final data without loaded metadata (df_full).")
            return

        print("\nPreparing final datasets using determined best parameters...")
        # Regenerate CNN1 data
        final_df_processed = apply_weighting_to_df(df_full, cnn1_dataprep_params['n_frames_weighting'], cnn1_dataprep_params['weight_decay'])
        final_cnn1_splits = balance_and_split_data(final_df_processed, cnn1_dataprep_params['balance_ratio'])
        cnn1_balanced_df_final = final_cnn1_splits[0]
        _, f_train_p1, f_train_t1, f_val_p1, f_val_t1, f_test_p1, f_test_t1 = final_cnn1_splits

        if not f_train_p1 or not f_val_p1 or not f_test_p1:
             print("ERROR: Final CNN1 data split failed. Check parameters and data.")
             return

        # Regenerate CNN2 data
        try:
             landing_df_indexed = load_landing_data(dataset_base_path)
             final_sequences_cnn2 = get_sequences_for_cnn2(cnn1_balanced_df_final, landing_df_indexed, cnn2_dataprep_params['n_frames_sequence_cnn2'])
             if not final_sequences_cnn2: raise ValueError("No CNN2 sequences generated.")
             f_train_seq2, f_val_seq2, f_test_seq2 = split_sequences(final_sequences_cnn2)
             final_cnn2_splits = (f_train_seq2, f_val_seq2, f_test_seq2) # Store splits

             if not f_train_seq2 or not f_val_seq2 or not f_test_seq2:
                 raise ValueError("Final CNN2 sequence split resulted in empty set(s).")

        except Exception as e:
             print(f"ERROR preparing final CNN2 data: {e}")
             return


        # Create Final DataLoaders (with Augmentation for Train)
        print("\nCreating final DataLoaders...")
        # CNN1
        cnn1_train_ds = TennisFrameDataset(f_train_p1, f_train_t1, augment=True)
        cnn1_val_ds = TennisFrameDataset(f_val_p1, f_val_t1, augment=False)
        cnn1_test_ds = TennisFrameDataset(f_test_p1, f_test_t1, augment=False)
        cnn1_train_loader = DataLoader(cnn1_train_ds, batch_size=cnn1_trainhp_params['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        cnn1_val_loader = DataLoader(cnn1_val_ds, batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        cnn1_test_loader = DataLoader(cnn1_test_ds, batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        print(f"CNN1 Loaders: Train={len(cnn1_train_loader)}, Val={len(cnn1_val_loader)}, Test={len(cnn1_test_loader)}")

        # CNN2
        cnn2_seq_len = cnn2_dataprep_params['n_frames_sequence_cnn2']
        cnn2_train_ds = BallLandingDataset(f_train_seq2, n_frames_sequence=cnn2_seq_len, augment=True)
        cnn2_val_ds = BallLandingDataset(f_val_seq2, n_frames_sequence=cnn2_seq_len, augment=False)
        cnn2_test_ds = BallLandingDataset(f_test_seq2, n_frames_sequence=cnn2_seq_len, augment=False)
        cnn2_train_loader = DataLoader(cnn2_train_ds, batch_size=cnn2_trainhp_params['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
        cnn2_val_loader = DataLoader(cnn2_val_ds, batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        cnn2_test_loader = DataLoader(cnn2_test_ds, batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        print(f"CNN2 Loaders: Train={len(cnn2_train_loader)}, Val={len(cnn2_val_loader)}, Test={len(cnn2_test_loader)}")

    # --- 6. Final Model Training (Optional) ---
    cnn1_history = None
    cnn2_history = None
    if args.run_final_training:
        if final_cnn1_splits is None or final_cnn2_splits is None:
             print("ERROR: Cannot run final training without final data splits.")
             return

        print("\n" + "#"*20 + " Starting Final Model Training " + "#"*20)
        gc.collect(); torch.cuda.empty_cache()

        # --- Train CNN1 ---
        print("\n--- Final Training: CNN1 (Hit Frame Regressor) ---")
        cnn1_model = HitFrameRegressorFinal(
            block_filters=tuple(cnn1_arch_params['filters']), # Ensure tuple
            fc_size=cnn1_arch_params['fc_size'],
            dropout_rate=cnn1_arch_params['dropout']
        ).to(config.DEVICE)
        criterion1 = nn.MSELoss()
        optimizer1 = optim.Adam(cnn1_model.parameters(), lr=cnn1_trainhp_params['learning_rate'])
        model1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
        history1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_history.csv')

        cnn1_history = train_model(
            model=cnn1_model, model_name="CNN1 Final", train_loader=cnn1_train_loader, val_loader=cnn1_val_loader,
            criterion=criterion1, optimizer=optimizer1, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS,
            early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE, min_improvement=config.DEFAULT_MIN_IMPROVEMENT,
            results_save_path=history1_path, best_model_save_path=model1_path, is_tuning_run=False
        )
        del cnn1_model, optimizer1, criterion1 # Cleanup
        gc.collect(); torch.cuda.empty_cache()

        # --- Train CNN2 ---
        print("\n--- Final Training: CNN2 (Landing Spot Predictor) ---")
        cnn2_input_channels = cnn2_dataprep_params['n_frames_sequence_cnn2'] * 3
        cnn2_model = LandingPointCNN(input_channels=cnn2_input_channels).to(config.DEVICE)
        criterion2 = nn.MSELoss()
        optimizer2 = optim.Adam(cnn2_model.parameters(), lr=cnn2_trainhp_params['learning_rate'])
        model2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_final.pth')
        history2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_final_training_history.csv')

        cnn2_history = train_model(
            model=cnn2_model, model_name="CNN2 Final", train_loader=cnn2_train_loader, val_loader=cnn2_val_loader,
            criterion=criterion2, optimizer=optimizer2, device=config.DEVICE, epochs=config.DEFAULT_FINAL_EPOCHS,
            early_stopping_patience=config.DEFAULT_EARLY_STOPPING_PATIENCE, min_improvement=config.DEFAULT_MIN_IMPROVEMENT,
            results_save_path=history2_path, best_model_save_path=model2_path, is_tuning_run=False
        )
        del cnn2_model, optimizer2, criterion2 # Cleanup
        gc.collect(); torch.cuda.empty_cache()

        print("\n" + "#"*20 + " Final Training Finished " + "#"*20)


    # --- 7. Evaluation (Optional) ---
    if args.run_evaluation:
        if final_cnn1_splits is None or final_cnn2_splits is None:
             print("ERROR: Cannot run evaluation without final data splits (regenerated or loaded).")
             print("Try running with --run_final_training first, or ensure splits can be recreated.")
             # Attempt to recreate splits if they weren't made (e.g. if only --run_evaluation is used)
             if df_full is not None and final_cnn1_splits is None:
                  print("Attempting to recreate splits for evaluation...")
                  # Repeat data prep steps from section 5 using loaded params
                  try:
                       final_df_processed = apply_weighting_to_df(df_full, cnn1_dataprep_params['n_frames_weighting'], cnn1_dataprep_params['weight_decay'])
                       final_cnn1_splits = balance_and_split_data(final_df_processed, cnn1_dataprep_params['balance_ratio'])
                       cnn1_balanced_df_final = final_cnn1_splits[0]
                       _, _, _, _, _, f_test_p1, f_test_t1 = final_cnn1_splits # Need test split

                       landing_df_indexed = load_landing_data(dataset_base_path)
                       final_sequences_cnn2 = get_sequences_for_cnn2(cnn1_balanced_df_final, landing_df_indexed, cnn2_dataprep_params['n_frames_sequence_cnn2'])
                       _, _, f_test_seq2 = split_sequences(final_sequences_cnn2) # Need test split

                       # Recreate test loaders
                       cnn1_test_ds = TennisFrameDataset(f_test_p1, f_test_t1, augment=False)
                       cnn1_test_loader = DataLoader(cnn1_test_ds, batch_size=cnn1_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

                       cnn2_seq_len = cnn2_dataprep_params['n_frames_sequence_cnn2']
                       cnn2_test_ds = BallLandingDataset(f_test_seq2, n_frames_sequence=cnn2_seq_len, augment=False)
                       cnn2_test_loader = DataLoader(cnn2_test_ds, batch_size=cnn2_trainhp_params['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
                       print("Recreated test splits and loaders successfully.")
                  except Exception as e:
                       print(f"ERROR: Failed to recreate splits for evaluation: {e}")
                       return
             elif final_cnn1_splits is None:
                 print("ERROR: Cannot evaluate without data splits and df_full.")
                 return


        print("\n" + "#"*20 + " Starting Evaluation " + "#"*20)
        model1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
        model2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_final.pth')

        # Evaluate CNN1
        cnn1_model_eval = load_final_cnn1_model(model1_path, config.DEVICE, cnn1_arch_params)
        if cnn1_model_eval and cnn1_test_loader:
            evaluate_model(cnn1_model_eval, "CNN1 Final", cnn1_test_loader, nn.MSELoss(), config.DEVICE)
            del cnn1_model_eval; gc.collect(); torch.cuda.empty_cache()
        else: print("Skipping CNN1 evaluation (model or loader error).")

        # Evaluate CNN2
        cnn2_input_channels = cnn2_dataprep_params['n_frames_sequence_cnn2'] * 3
        cnn2_model_eval = load_final_cnn2_model(model2_path, config.DEVICE, cnn2_input_channels)
        if cnn2_model_eval and cnn2_test_loader:
            evaluate_model(cnn2_model_eval, "CNN2 Final", cnn2_test_loader, nn.MSELoss(), config.DEVICE)
            del cnn2_model_eval; gc.collect(); torch.cuda.empty_cache()
        else: print("Skipping CNN2 evaluation (model or loader error).")

        # Plot Histories
        print("\nPlotting training histories...")
        history1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_history.csv')
        plot_path1 = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_final_training_plot.png')
        plot_training_history(cnn1_history or history1_path, "CNN1 (Hit Frame Regressor)", plot_path1)

        history2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_final_training_history.csv')
        plot_path2 = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_final_training_plot.png')
        plot_training_history(cnn2_history or history2_path, "CNN2 (Landing Spot Predictor)", plot_path2)

        print("\n" + "#"*20 + " Evaluation Finished " + "#"*20)

    # --- 8. Prediction Pipeline Example (Optional) ---
    if args.run_prediction:
        print("\n" + "#"*20 + " Running Prediction Example " + "#"*20)
        model1_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_hit_frame_regressor_final.pth')
        model2_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_landing_spot_predictor_final.pth')

        # Load models needed for prediction
        cnn1_pred_model = load_final_cnn1_model(model1_path, config.DEVICE, cnn1_arch_params)
        cnn2_input_channels = cnn2_dataprep_params['n_frames_sequence_cnn2'] * 3
        cnn2_pred_model = load_final_cnn2_model(model2_path, config.DEVICE, cnn2_input_channels)

        if not cnn1_pred_model or not cnn2_pred_model:
            print("Error: Cannot run prediction example, models not loaded.")
        else:
            example_dir = None
            actual_coords_example = None
            # Try to get an example from test data if available
            if final_cnn2_splits:
                 test_sequences = final_cnn2_splits[2]
                 if test_sequences:
                      example_sequence_info = random.choice(test_sequences)
                      example_dir = os.path.dirname(example_sequence_info['sequence_paths'][0])
                      actual_coords_example = example_sequence_info['target_coords']
                      print(f"Selected example directory: {os.path.basename(example_dir)}")
                      if actual_coords_example: print(f"Actual Landing Coords: ({actual_coords_example[0]:.3f}, {actual_coords_example[1]:.3f})")
                 else: print("Test sequence list is empty, cannot select example automatically.")

            # Fallback: Manually specify a directory if needed
            if not example_dir and dataset_base_path:
                 # Try to find *any* shot directory
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
                    cnn2_seq_len=cnn2_dataprep_params['n_frames_sequence_cnn2'],
                    device=config.DEVICE
                )

                if predicted_coords and predicted_hit_path:
                    print("\n--- Example Pipeline Results ---")
                    print(f"Directory: {os.path.basename(example_dir)}")
                    print(f"Predicted Hit Frame: {os.path.basename(predicted_hit_path)}")
                    print(f"Predicted Landing (Norm): ({predicted_coords[0]:.4f}, {predicted_coords[1]:.4f})")

                    dl, dr, db = denormalize_coordinates(predicted_coords[0], predicted_coords[1])
                    print(f"Predicted Landing (Approx. Meters): L={dl:.2f}m, R={dr:.2f}m, Baseline={db:.2f}m")

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

            else: print("Could not find a valid example directory to run prediction.")

        del cnn1_pred_model, cnn2_pred_model # Cleanup models
        gc.collect(); torch.cuda.empty_cache()


    end_time_main = time.time()
    print(f"\nTotal execution time: {end_time_main - start_time_main:.2f} seconds.")
    print("="*50)
    print(" Pipeline Finished ".center(50, "="))
    print("="*50)


# --- Entry Point ---
if __name__ == "__main__":
    # Needed for Windows multiprocessing in DataLoader with num_workers > 0
    # Should be at the top level of the script execution block
    torch.multiprocessing.set_start_method('spawn') if config.DEVICE.type == 'cuda' and os.name == 'nt' else None # Or 'forkserver' on Linux if needed

    parser = argparse.ArgumentParser(description="ST311 Tennis Analysis Pipeline")
    parser.add_argument('--download_data', action='store_true', help='Attempt to download dataset from Kaggle.')
    parser.add_argument('--run_grid_search', action='store_true', help='Run hyperparameter grid search phases.')
    parser.add_argument('--run_final_training', action='store_true', help='Run final training using best/default parameters.')
    parser.add_argument('--run_evaluation', action='store_true', help='Evaluate final models on the test set.')
    parser.add_argument('--run_prediction', action='store_true', help='Run the prediction pipeline on an example.')
    parser.add_argument('--all', action='store_true', help='Run all steps (download, grid search, training, evaluation, prediction).')

    args = parser.parse_args()

    # If --all is specified, set all flags to True
    if args.all:
        args.download_data = True
        args.run_grid_search = True
        args.run_final_training = True
        args.run_evaluation = True
        args.run_prediction = True

    # Check if at least one action is requested
    if not any([args.download_data, args.run_grid_search, args.run_final_training, args.run_evaluation, args.run_prediction]):
        print("No action specified. Use --help to see options.")
        print("Example usage:")
        print("  python main.py --all")
        print("  python main.py --run_final_training --run_evaluation")
        print("  python main.py --run_prediction")
    else:
        main(args)