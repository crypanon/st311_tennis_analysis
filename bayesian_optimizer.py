# -*- coding: utf-8 -*-
import optuna
# ... (other imports: torch, nn, optim, DataLoader, os, gc, json, math) ...
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc
import json
import math
import os
import config
from data_utils import apply_bayesian_weighting_to_df, balance_and_split_data # Keep these
from datasets import TennisFrameDataset
from models import HitFrameRegressorFinal
from training import train_model

# Objective function for Optuna to minimize (CNN1 validation loss)
def cnn1_objective(trial: optuna.Trial, df_full, cnn1_arch_params, device):
    """Optuna objective function for CNN1 weighting parameters (h(x) version)."""
    gc.collect(); torch.cuda.empty_cache()

    # 1. Suggest parameters for h(x)
    # Use suggest_int for R1, R2
    R1 = trial.suggest_int("R1", *config.BAYESIAN_PARAM_RANGES["R1"])
    R2 = trial.suggest_int("R2", *config.BAYESIAN_PARAM_RANGES["R2"])
    N = trial.suggest_float("N", *config.BAYESIAN_PARAM_RANGES["N"])
    D = trial.suggest_float("D", *config.BAYESIAN_PARAM_RANGES["D"])
    M1 = trial.suggest_float("M1", *config.BAYESIAN_PARAM_RANGES["M1"])
    M2 = trial.suggest_float("M2", *config.BAYESIAN_PARAM_RANGES["M2"])
    balance_ratio = config.DEFAULT_BALANCE_RATIO

    print(f"\nTrial {trial.number}: R1={R1}, R2={R2}, N={N:.3f}, D={D:.3f}, M1={M1:.3f}, M2={M2:.3f}")

    try:
        # 2. Apply weighting (using full h(x)) and split data
        df_weighted = apply_bayesian_weighting_to_df(df_full, R1, R2, N, D, M1, M2)
        splits = balance_and_split_data(df_weighted, balance_ratio)
        _, train_p, train_t, val_p, val_t, _, _ = splits

        if not train_p or not val_p:
            print("WARN: Empty train/val split in trial. Returning infinity.")
            return float('inf')

        # 3. Create DataLoaders
        train_ds = TennisFrameDataset(train_p, train_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
        val_ds = TennisFrameDataset(val_p, val_t, config.IMG_HEIGHT, config.IMG_WIDTH, augment=False)
        train_loader = DataLoader(train_ds, batch_size=config.DEFAULT_CNN1_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        val_loader = DataLoader(val_ds, batch_size=config.DEFAULT_CNN1_BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

        # 4. Instantiate Model (best ARCHITECTURE)
        filters, fc_size, dropout = cnn1_arch_params
        model = HitFrameRegressorFinal(block_filters=filters, fc_size=fc_size, dropout_rate=dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.DEFAULT_CNN1_LR)
        criterion = nn.MSELoss() # Still use MSE loss to match CNN1 output against h(x) targets

        # 5. Train briefly
        final_val_loss = train_model(
            model=model, model_name=f"CNN1 (BayesOpt Trial {trial.number})",
            train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            epochs=config.BAYESIAN_OPT_TUNING_EPOCHS, is_tuning_run=True, early_stopping_patience=0
        )

        if not math.isfinite(final_val_loss):
             print("WARN: Non-finite validation loss. Returning infinity.")
             final_val_loss = float('inf')

        print(f"Trial {trial.number} finished. Val Loss: {final_val_loss:.6f}")
        return final_val_loss

    except Exception as e:
        print(f"!!! ERROR in Trial {trial.number}: {e} !!!")
        return float('inf')
    finally:
         del model, optimizer, train_loader, val_loader, train_ds, val_ds, df_weighted
         gc.collect(); torch.cuda.empty_cache()

def run_bayesian_optimization(df_full, cnn1_arch_params, device):
    """Runs the Optuna study for h(x) parameters."""
    print("\n" + "="*30 + " CNN1 Bayesian Optimization for h(x) Weighting " + "="*30)
    study = optuna.create_study(direction="minimize", study_name="CNN1 h(x) Weighting Opt")
    objective_func = lambda trial: cnn1_objective(trial, df_full, cnn1_arch_params, device)
    study.optimize(objective_func, n_trials=config.BAYESIAN_OPT_N_TRIALS)

    print("\n--- Bayesian Optimization Finished ---")
    # ... (print results and save best params - unchanged) ...
    best_params = study.best_trial.params
    # Store integer R1/R2 globally for joint training loop
    config.OPTIMIZED_R1_INT = int(round(best_params['R1'])) # Round to nearest int
    config.OPTIMIZED_R2_INT = int(round(best_params['R2']))
    print(f"Optimized R1 (int): {config.OPTIMIZED_R1_INT}, R2 (int): {config.OPTIMIZED_R2_INT}")

    best_params_path = os.path.join(config.PROJECT_OUTPUT_PATH, 'best_cnn1_bayesian_weights.json')
    with open(best_params_path, 'w') as f: json.dump(best_params, f, indent=4)
    print(f"Best raw weighting parameters saved to: {best_params_path}")

    return best_params # Return the raw best params dict