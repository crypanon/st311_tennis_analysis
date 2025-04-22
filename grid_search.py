# -*- coding: utf-8 -*-
import os
import itertools
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import json
import math

# Import project modules
import config
from models import HitFrameRegressorParam, HitFrameRegressorFinal, LandingPointCNNParam, LandingPointCNN
from datasets import TennisFrameDataset, BallLandingDataset
from training import train_model
from data_utils import apply_linear_weighting_to_df, balance_and_split_data, get_sequences_for_cnn2, split_sequences

def run_cnn1_arch_search(df_full, initial_splits, device):
    """Performs grid search for CNN1 architecture."""
    print("\n" + "="*30 + " CNN1 Architecture Search " + "="*30)
    print(f"Epochs per trial: {config.GRID_SEARCH_TUNING_EPOCHS}")
    print(f"Max candidates: {config.GRID_SEARCH_ARCHITECTURE_CANDIDATES}")

    # --- Define Architecture Options ---
    filter_options = [
        (32, 64, 128), (32, 64, 128, 256), (64, 128, 256), (32, 32, 64, 64),
        (48, 96, 192), (64, 128, 128), (32, 64), (16, 32, 64, 128)
    ]
    fc_size_options = [256, 512, 1024]
    dropout_options = [0.5] # Fixed for this search

    combinations = list(itertools.product(filter_options, fc_size_options, dropout_options))
    random.shuffle(combinations)
    if config.GRID_SEARCH_ARCHITECTURE_CANDIDATES > 0:
        combinations = combinations[:config.GRID_SEARCH_ARCHITECTURE_CANDIDATES]
    print(f"Testing {len(combinations)} architecture combinations.")

    # Use initial data splits for this search
    _, train_p, train_t, val_p, val_t, _, _ = initial_splits
    if not train_p or not val_p:
        print("ERROR: Initial data splits are empty. Cannot run arch search.")
        return config.DEFAULT_CNN1_FILTERS, config.DEFAULT_CNN1_FC_SIZE, config.DEFAULT_CNN1_DROPOUT # Return defaults

    temp_train_ds = TennisFrameDataset(train_p, train_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
    temp_val_ds = TennisFrameDataset(val_p, val_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
    # Use default BS/LR from config for this specific search
    temp_train_loader = DataLoader(temp_train_ds, batch_size=config.DEFAULT_CNN1_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    temp_val_loader = DataLoader(temp_val_ds, batch_size=config.DEFAULT_CNN1_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    temp_criterion = nn.MSELoss()
    temp_lr = config.DEFAULT_CNN1_LR

    results = []
    best_val_loss = float('inf')
    best_params = None

    for i, (filters, fc_size, dropout) in enumerate(combinations):
        print(f"\n--- Arch Candidate {i+1}/{len(combinations)}: Filters={filters}, FC={fc_size}, Dropout={dropout} ---")
        gc.collect(); torch.cuda.empty_cache()

        try:
            model = HitFrameRegressorParam(
                block_filters=filters, fc_size=fc_size, dropout_rate=dropout
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=temp_lr)

            val_loss = train_model(
                model=model, model_name="CNN1 (Arch Tuning)", train_loader=temp_train_loader, val_loader=temp_val_loader,
                criterion=temp_criterion, optimizer=optimizer, device=device, epochs=config.GRID_SEARCH_TUNING_EPOCHS,
                is_tuning_run=True, early_stopping_patience=0 # No ES for tuning runs
            )

            results.append({'filters': str(filters), 'fc_size': fc_size, 'dropout': dropout, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'filters': filters, 'fc_size': fc_size, 'dropout': dropout}
                print(f"*** New best arch val_loss: {best_val_loss:.6f} ***")

        except Exception as e:
            print(f"!!! ERROR training arch candidate {i+1}: {e} !!!")
            results.append({'filters': str(filters), 'fc_size': fc_size, 'dropout': dropout, 'val_loss': float('inf'), 'error': str(e)})
        finally:
             del model, optimizer # Explicit cleanup
             gc.collect(); torch.cuda.empty_cache()


    # Save results and best params
    results_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_architecture_search_results.csv')
    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_architecture.json')

    if results:
        df_results = pd.DataFrame(results).sort_values('val_loss')
        df_results.to_csv(results_path, index=False)
        print(f"Arch search results saved to: {results_path}")

    if best_params:
        print(f"\nBest CNN1 Architecture Found: {best_params} (Val Loss: {best_val_loss:.6f})")
        # Save (convert tuple filters to list for JSON)
        best_params_save = best_params.copy()
        best_params_save['filters'] = list(best_params_save['filters'])
        with open(best_params_path, 'w') as f: json.dump(best_params_save, f, indent=4)
        print(f"Best arch params saved to: {best_params_path}")
        return best_params['filters'], best_params['fc_size'], best_params['dropout']
    else:
        print("Arch search failed to find a best model. Using defaults.")
        return config.DEFAULT_CNN1_FILTERS, config.DEFAULT_CNN1_FC_SIZE, config.DEFAULT_CNN1_DROPOUT


def run_cnn1_dataprep_search(df_full, best_arch_params, device):
    """Performs grid search for CNN1 data prep parameters."""
    print("\n" + "="*30 + " CNN1 Data Prep Search " + "="*30)
    print(f"Epochs per trial: {config.GRID_SEARCH_TUNING_EPOCHS}")

    # --- Define Data Prep Options ---
    n_frames_options = [7, 9, 11]
    decay_options = [0.15, 0.3, 0.45]
    balance_options = [2, 4, 6]

    combinations = list(itertools.product(n_frames_options, decay_options, balance_options))
    random.shuffle(combinations)
    print(f"Testing {len(combinations)} data prep combinations.")

    # Use best arch found previously
    filters, fc_size, dropout = best_arch_params
    temp_criterion = nn.MSELoss()
    temp_lr = config.DEFAULT_CNN1_LR # Use default LR/BS
    temp_bs = config.DEFAULT_CNN1_BATCH_SIZE

    results = []
    best_val_loss = float('inf')
    best_params = None
    best_final_splits = None # Store the splits corresponding to the best params

    for i, (n_frames, decay, balance) in enumerate(combinations):
        print(f"\n--- DataPrep Candidate {i+1}/{len(combinations)}: N_Frames={n_frames}, Decay={decay}, Balance={balance} ---")
        gc.collect(); torch.cuda.empty_cache()

        try:
            # 1. Regenerate data with current params
            current_df_processed = apply_linear_weighting_to_df(df_full, n_frames, decay)
            current_splits = balance_and_split_data(current_df_processed, balance)
            _, train_p, train_t, val_p, val_t, _, _ = current_splits

            if not train_p or not val_p:
                print("WARN: Empty train/val split. Skipping.")
                continue

            # 2. Create DataLoaders
            train_ds = TennisFrameDataset(train_p, train_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
            val_ds = TennisFrameDataset(val_p, val_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
            train_loader = DataLoader(train_ds, batch_size=temp_bs, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
            val_loader = DataLoader(val_ds, batch_size=temp_bs, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

            # 3. Instantiate Model with best arch
            model = HitFrameRegressorFinal(block_filters=filters, fc_size=fc_size, dropout_rate=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=temp_lr)

            # 4. Train briefly
            val_loss = train_model(
                model=model, model_name="CNN1 (DataPrep Tuning)", train_loader=train_loader, val_loader=val_loader,
                criterion=temp_criterion, optimizer=optimizer, device=device, epochs=config.GRID_SEARCH_TUNING_EPOCHS,
                is_tuning_run=True, early_stopping_patience=0
            )

            results.append({'n_frames': n_frames, 'decay': decay, 'balance': balance, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'n_frames_weighting': n_frames, 'weight_decay': decay, 'balance_ratio': balance}
                best_final_splits = current_splits # Save the data splits too
                print(f"*** New best data prep val_loss: {best_val_loss:.6f} ***")

        except Exception as e:
            print(f"!!! ERROR training dataprep candidate {i+1}: {e} !!!")
            results.append({'n_frames': n_frames, 'decay': decay, 'balance': balance, 'val_loss': float('inf'), 'error': str(e)})
        finally:
            # Explicit cleanup of potentially large objects
            del model, optimizer, train_loader, val_loader, train_ds, val_ds, current_df_processed
            gc.collect(); torch.cuda.empty_cache()

    # Save results and best params
    results_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_dataprep_search_results.csv')
    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_dataprep.json')

    if results:
        df_results = pd.DataFrame(results).sort_values('val_loss')
        df_results.to_csv(results_path, index=False)
        print(f"Data prep search results saved to: {results_path}")

    if best_params and best_final_splits:
        print(f"\nBest CNN1 Data Prep Found: {best_params} (Val Loss: {best_val_loss:.6f})")
        with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
        print(f"Best data prep params saved to: {best_params_path}")
        # Return best params AND the corresponding data split
        return best_params['n_frames_weighting'], best_params['weight_decay'], best_params['balance_ratio'], best_final_splits
    else:
        print("Data prep search failed. Using defaults and initial splits.")
        # Need to return something reasonable for splits, maybe re-run with defaults?
        # For simplicity, let's assume initial splits are okay if search fails, but this is suboptimal.
        print("WARNING: Falling back to initial data splits for subsequent steps.")
        initial_splits = balance_and_split_data(df_full, config.DEFAULT_BALANCE_RATIO) # Use weighted df_full? No, raw df_full
        # Re-run weighting/balancing with defaults
        df_processed_default = apply_linear_weighting_to_df(df_full, config.DEFAULT_N_FRAMES_WEIGHTING, config.DEFAULT_WEIGHT_DECAY)
        default_splits = balance_and_split_data(df_processed_default, config.DEFAULT_BALANCE_RATIO)

        return config.DEFAULT_N_FRAMES_WEIGHTING, config.DEFAULT_WEIGHT_DECAY, config.DEFAULT_BALANCE_RATIO, default_splits


def run_cnn1_trainhp_search(final_cnn1_splits, best_arch_params, device):
    """Performs grid search for CNN1 LR and Batch Size."""
    print("\n" + "="*30 + " CNN1 Training HP Search " + "="*30)
    print(f"Epochs per trial: {config.GRID_SEARCH_TUNING_EPOCHS}")

    lr_options = [1e-3, 5e-4, 1e-4, 5e-5]
    bs_options = [16, 32, 64] # Adjust based on GPU memory

    combinations = list(itertools.product(lr_options, bs_options))
    random.shuffle(combinations)
    print(f"Testing {len(combinations)} training HP combinations.")

    # Use final data splits determined by data prep search
    _, train_p, train_t, val_p, val_t, _, _ = final_cnn1_splits
    if not train_p or not val_p:
        print("ERROR: Final CNN1 data splits are empty. Cannot run train HP search.")
        return config.DEFAULT_CNN1_LR, config.DEFAULT_CNN1_BATCH_SIZE

    # Create datasets ONCE
    train_ds = TennisFrameDataset(train_p, train_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
    val_ds = TennisFrameDataset(val_p, val_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)

    filters, fc_size, dropout = best_arch_params
    temp_criterion = nn.MSELoss()

    results = []
    best_val_loss = float('inf')
    best_params = None

    for i, (lr, batch_size) in enumerate(combinations):
        print(f"\n--- Train HP Candidate {i+1}/{len(combinations)}: LR={lr}, BS={batch_size} ---")
        gc.collect(); torch.cuda.empty_cache()

        try:
            # Create DataLoaders with current batch size
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
            if len(train_loader) == 0 or len(val_loader) == 0:
                 print(f"WARN: DataLoader empty for BS={batch_size}. Skipping.")
                 continue

            model = HitFrameRegressorFinal(block_filters=filters, fc_size=fc_size, dropout_rate=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            val_loss = train_model(
                model=model, model_name="CNN1 (TrainHP Tuning)", train_loader=train_loader, val_loader=val_loader,
                criterion=temp_criterion, optimizer=optimizer, device=device, epochs=config.GRID_SEARCH_TUNING_EPOCHS,
                is_tuning_run=True, early_stopping_patience=0
            )

            results.append({'lr': lr, 'batch_size': batch_size, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                print(f"*** New best train HP val_loss: {best_val_loss:.6f} ***")

        except Exception as e:
            print(f"!!! ERROR training train HP candidate {i+1}: {e} !!!")
            results.append({'lr': lr, 'batch_size': batch_size, 'val_loss': float('inf'), 'error': str(e)})
        finally:
            del model, optimizer, train_loader, val_loader # Cleanup loaders too
            gc.collect(); torch.cuda.empty_cache()


    results_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn1_training_hp_search_results.csv')
    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_training_hp.json')

    if results:
        df_results = pd.DataFrame(results).sort_values('val_loss')
        df_results.to_csv(results_path, index=False)
        print(f"Train HP search results saved to: {results_path}")

    if best_params:
        print(f"\nBest CNN1 Training HP Found: {best_params} (Val Loss: {best_val_loss:.6f})")
        with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
        print(f"Best train HP params saved to: {best_params_path}")
        return best_params['learning_rate'], best_params['batch_size']
    else:
        print("Train HP search failed. Using defaults.")
        return config.DEFAULT_CNN1_LR, config.DEFAULT_CNN1_BATCH_SIZE


# --- CNN2 Searches ---

def run_cnn2_dataprep_search(cnn1_balanced_df, landing_df, device):
    """Performs grid search for CNN2 sequence length."""
    print("\n" + "="*30 + " CNN2 Data Prep (Seq Len) Search " + "="*30)
    print(f"Epochs per trial: {config.GRID_SEARCH_TUNING_EPOCHS}")

    seq_len_options = [5, 7, 9] # Must be odd
    print(f"Testing {len(seq_len_options)} sequence lengths.")

    # Use defaults for LR/BS during this search
    temp_lr = config.DEFAULT_CNN2_LR
    temp_bs = config.DEFAULT_CNN2_BATCH_SIZE
    temp_criterion = nn.MSELoss()

    results = []
    best_val_loss = float('inf')
    best_params = None
    best_final_sequences = None # Store sequences for the best length

    for i, seq_len in enumerate(seq_len_options):
        print(f"\n--- SeqLen Candidate {i+1}/{len(seq_len_options)}: N_Frames={seq_len} ---")
        current_input_channels = seq_len * 3
        gc.collect(); torch.cuda.empty_cache()

        try:
            # 1. Regenerate sequences and splits
            current_sequences = get_sequences_for_cnn2(cnn1_balanced_df, landing_df, seq_len)
            if not current_sequences:
                print("WARN: No sequences generated. Skipping.")
                continue
            train_seq, val_seq, _ = split_sequences(current_sequences)
            if not train_seq or not val_seq:
                print("WARN: Empty train/val split for sequences. Skipping.")
                continue

            # 2. Create DataLoaders
            train_ds = BallLandingDataset(train_seq, config.IMG_HEIGHT, config.IMG_WIDTH, seq_len, augment=False)
            val_ds = BallLandingDataset(val_seq, config.IMG_HEIGHT, config.IMG_WIDTH, seq_len, augment=False)
            train_loader = DataLoader(train_ds, batch_size=temp_bs, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
            val_loader = DataLoader(val_ds, batch_size=temp_bs, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

            # 3. Instantiate CNN2 Model
            model = LandingPointCNN(input_channels=current_input_channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=temp_lr)

            # 4. Train briefly
            val_loss = train_model(
                model=model, model_name="CNN2 (SeqLen Tuning)", train_loader=train_loader, val_loader=val_loader,
                criterion=temp_criterion, optimizer=optimizer, device=device, epochs=config.GRID_SEARCH_TUNING_EPOCHS,
                is_tuning_run=True, early_stopping_patience=0
            )

            results.append({'seq_len': seq_len, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'n_frames_sequence_cnn2': seq_len}
                best_final_sequences = (train_seq, val_seq, _) # Save corresponding sequences
                print(f"*** New best CNN2 SeqLen val_loss: {best_val_loss:.6f} ***")

        except Exception as e:
            print(f"!!! ERROR training SeqLen candidate {i+1}: {e} !!!")
            results.append({'seq_len': seq_len, 'val_loss': float('inf'), 'error': str(e)})
        finally:
            del model, optimizer, train_loader, val_loader, train_ds, val_ds, current_sequences
            gc.collect(); torch.cuda.empty_cache()

    results_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_dataprep_search_results.csv')
    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_dataprep.json')

    if results:
        df_results = pd.DataFrame(results).sort_values('val_loss')
        df_results.to_csv(results_path, index=False)
        print(f"CNN2 SeqLen search results saved to: {results_path}")

    if best_params and best_final_sequences:
        print(f"\nBest CNN2 Sequence Length Found: {best_params} (Val Loss: {best_val_loss:.6f})")
        with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
        print(f"Best CNN2 SeqLen params saved to: {best_params_path}")
        return best_params['n_frames_sequence_cnn2'], best_final_sequences
    else:
        print("CNN2 SeqLen search failed. Using defaults.")
        # Regenerate sequences with default length
        default_sequences = get_sequences_for_cnn2(cnn1_balanced_df, landing_df, config.DEFAULT_N_FRAMES_SEQUENCE_CNN2)
        default_seq_splits = split_sequences(default_sequences)
        return config.DEFAULT_N_FRAMES_SEQUENCE_CNN2, default_seq_splits


def run_cnn2_trainhp_search(final_cnn2_splits, best_seq_len, device):
    """Performs grid search for CNN2 LR and Batch Size."""
    print("\n" + "="*30 + " CNN2 Training HP Search " + "="*30)
    print(f"Epochs per trial: {config.GRID_SEARCH_TUNING_EPOCHS}")

    lr_options = [1e-4, 5e-5, 1e-5, 5e-6]
    bs_options = [8, 16, 32] # Often smaller for CNN2 due to larger input tensor

    combinations = list(itertools.product(lr_options, bs_options))
    random.shuffle(combinations)
    print(f"Testing {len(combinations)} CNN2 training HP combinations.")

    # Use final sequence splits
    train_seq, val_seq, _ = final_cnn2_splits
    if not train_seq or not val_seq:
        print("ERROR: Final CNN2 sequence splits are empty. Cannot run train HP search.")
        return config.DEFAULT_CNN2_LR, config.DEFAULT_CNN2_BATCH_SIZE

    # Create datasets ONCE
    train_ds = BallLandingDataset(train_seq, config.IMG_HEIGHT, config.IMG_WIDTH, best_seq_len, augment=False)
    val_ds = BallLandingDataset(val_seq, config.IMG_HEIGHT, config.IMG_WIDTH, best_seq_len, augment=False)

    input_channels = best_seq_len * 3
    temp_criterion = nn.MSELoss()

    results = []
    best_val_loss = float('inf')
    best_params = None

    for i, (lr, batch_size) in enumerate(combinations):
        print(f"\n--- CNN2 Train HP Candidate {i+1}/{len(combinations)}: LR={lr}, BS={batch_size} ---")
        gc.collect(); torch.cuda.empty_cache()

        try:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
            if len(train_loader) == 0 or len(val_loader) == 0:
                 print(f"WARN: DataLoader empty for BS={batch_size}. Skipping.")
                 continue

            model = LandingPointCNN(input_channels=input_channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            val_loss = train_model(
                model=model, model_name="CNN2 (TrainHP Tuning)", train_loader=train_loader, val_loader=val_loader,
                criterion=temp_criterion, optimizer=optimizer, device=device, epochs=config.GRID_SEARCH_TUNING_EPOCHS,
                is_tuning_run=True, early_stopping_patience=0
            )

            results.append({'lr': lr, 'batch_size': batch_size, 'val_loss': val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                print(f"*** New best CNN2 train HP val_loss: {best_val_loss:.6f} ***")

        except Exception as e:
            print(f"!!! ERROR training CNN2 train HP candidate {i+1}: {e} !!!")
            results.append({'lr': lr, 'batch_size': batch_size, 'val_loss': float('inf'), 'error': str(e)})
        finally:
            del model, optimizer, train_loader, val_loader
            gc.collect(); torch.cuda.empty_cache()

    results_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_training_hp_search_results.csv')
    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_training_hp.json')

    if results:
        df_results = pd.DataFrame(results).sort_values('val_loss')
        df_results.to_csv(results_path, index=False)
        print(f"CNN2 Train HP search results saved to: {results_path}")

    if best_params:
        print(f"\nBest CNN2 Training HP Found: {best_params} (Val Loss: {best_val_loss:.6f})")
        with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
        print(f"Best CNN2 train HP params saved to: {best_params_path}")
        return best_params['learning_rate'], best_params['batch_size']
    else:
        print("CNN2 Train HP search failed. Using defaults.")
        return config.DEFAULT_CNN2_LR, config.DEFAULT_CNN2_BATCH_SIZE

# --- NEW: CNN2 Architecture Search ---
def run_cnn2_arch_search(standard_cnn2_splits, standard_cnn2_seq_len, device):
    """Performs grid search for CNN2 architecture."""
    print("\n" + "="*30 + " CNN2 Architecture Search " + "="*30)
    print(f"Epochs per trial: {config.GRID_SEARCH_TUNING_EPOCHS}")
    print(f"Using fixed sequence length for search: {standard_cnn2_seq_len}")

    # --- Define Architecture Options ---
    # Example: Vary filters in the original 4 blocks
    cnn2_filter_options = [
        (64, 128, 256, 512), # Original
        (32, 64, 128, 256),  # Slimmer
        (64, 128, 256),      # Shallower (3 blocks)
        (128, 256, 512, 512),# Wider later
    ]
    # Example: Vary FC layers
    cnn2_fc_size_options = [
        (1024, 512),         # Original
        (512, 256),          # Smaller FC
        (1024,),             # Single large FC layer
    ]
    # Keep dropout fixed for this search
    cnn2_dropout_options = [config.DEFAULT_CNN2_DROPOUT]

    combinations = list(itertools.product(cnn2_filter_options, cnn2_fc_size_options, cnn2_dropout_options))
    random.shuffle(combinations)
    # Limit candidates if needed (can add a config param like CNN1)
    # combinations = combinations[:MAX_CNN2_ARCH_CANDIDATES]
    print(f"Testing {len(combinations)} CNN2 architecture combinations.")

    # Use standard CNN2 splits and sequence length for this search
    train_seq, val_seq, _ = standard_cnn2_splits
    if not train_seq or not val_seq:
        print("ERROR: Standard CNN2 data splits are empty. Cannot run arch search.")
        return config.DEFAULT_CNN2_CONV_FILTERS, config.DEFAULT_CNN2_FC_SIZES, config.DEFAULT_CNN2_DROPOUT # Return defaults

    # Create datasets ONCE using the standard sequence length
    input_channels = standard_cnn2_seq_len * 3
    train_ds = BallLandingDataset(train_seq, config.IMG_HEIGHT, config.IMG_WIDTH, standard_cnn2_seq_len, augment=False)
    val_ds = BallLandingDataset(val_seq, config.IMG_HEIGHT, config.IMG_WIDTH, standard_cnn2_seq_len, augment=False)

    # Use default/fixed CNN2 LR/BS for this architecture search
    temp_lr = config.DEFAULT_CNN2_LR
    temp_bs = config.DEFAULT_CNN2_BATCH_SIZE
    temp_criterion = nn.MSELoss()

    results = []
    best_val_loss = float('inf')
    best_params = None

    for i, (conv_filters, fc_sizes, dropout) in enumerate(combinations):
        print(f"\n--- CNN2 Arch Cand {i+1}/{len(combinations)}: Conv={conv_filters}, FC={fc_sizes}, Drop={dropout} ---")
        gc.collect(); torch.cuda.empty_cache()

        try:
            # Instantiate PARAMETERIZED CNN2 model
            model = LandingPointCNNParam(
                input_channels=input_channels, # Fixed for search
                conv_filters=conv_filters,
                fc_sizes=fc_sizes,
                dropout_rate=dropout
            ).to(device)

            # Create DataLoaders (batch size fixed for arch search for now)
            train_loader = DataLoader(train_ds, batch_size=temp_bs, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True)
            val_loader = DataLoader(val_ds, batch_size=temp_bs, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
            if len(train_loader) == 0 or len(val_loader) == 0: continue # Skip if BS too large

            optimizer = optim.Adam(model.parameters(), lr=temp_lr)

            val_loss = train_model(
                model=model, model_name="CNN2 (Arch Tuning)", train_loader=train_loader, val_loader=val_loader,
                criterion=temp_criterion, optimizer=optimizer, device=device, epochs=config.GRID_SEARCH_TUNING_EPOCHS,
                is_tuning_run=True, early_stopping_patience=0
            )

            if not math.isfinite(val_loss): val_loss = float('inf') # Handle potential errors
            results.append({'conv_filters': str(conv_filters), 'fc_sizes': str(fc_sizes), 'dropout': dropout, 'val_loss': val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {'conv_filters': conv_filters, 'fc_sizes': fc_sizes, 'dropout': dropout}
                print(f"*** New best CNN2 arch val_loss: {best_val_loss:.6f} ***")

        except Exception as e:
            print(f"!!! ERROR training CNN2 arch candidate {i+1}: {e} !!!")
            results.append({'conv_filters': str(conv_filters), 'fc_sizes': str(fc_sizes), 'dropout': dropout, 'val_loss': float('inf'), 'error': str(e)})
        finally:
             # Explicit cleanup
             del model, optimizer, train_loader, val_loader
             gc.collect(); torch.cuda.empty_cache()


    # Save results and best params
    results_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'cnn2_architecture_search_results.csv')
    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn2_architecture.json')

    if results:
        df_results = pd.DataFrame(results).sort_values('val_loss')
        df_results.to_csv(results_path, index=False)
        print(f"CNN2 Arch search results saved to: {results_path}")

    if best_params:
        print(f"\nBest CNN2 Architecture Found: {best_params} (Val Loss: {best_val_loss:.6f})")
        # Save (convert tuples to lists for JSON)
        best_params_save = best_params.copy()
        best_params_save['conv_filters'] = list(best_params_save['conv_filters'])
        best_params_save['fc_sizes'] = list(best_params_save['fc_sizes'])
        with open(best_params_path, 'w') as f: json.dump(best_params_save, f, indent=4)
        print(f"Best CNN2 arch params saved to: {best_params_path}")
        # Return parameters needed for model instantiation
        return best_params['conv_filters'], best_params['fc_sizes'], best_params['dropout']
    else:
        print("CNN2 Arch search failed to find a best model. Using defaults.")
        return config.DEFAULT_CNN2_CONV_FILTERS, config.DEFAULT_CNN2_FC_SIZES, config.DEFAULT_CNN2_DROPOUT