# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from tqdm import tqdm # Use standard tqdm
import math
import config

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, device, epochs,
                early_stopping_patience=10, min_improvement=1e-5,
                results_save_path=None, best_model_save_path=None, is_tuning_run=False):
    """Trains/validates model, supports early stopping, saves best model/history."""
    start_time = time.time()
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    # Ensure save directories exist
    if best_model_save_path: os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
    if results_save_path: os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    log_level = "INFO" if not is_tuning_run else "WARNING" # Reduce verbosity for tuning
    print(f"--- Starting Training ({model_name}) for {epochs} epochs ---", flush=True)
    if early_stopping_patience > 0:
        print(f"Early stopping: Patience={early_stopping_patience}, Min Improvement={min_improvement}", flush=True)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        # Use non_blocking for potentially faster transfer if pin_memory=True
        non_blocking = device.type == 'cuda'
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, ncols=80)

        for inputs, targets in train_loop:
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            # Adjust target shape (ensure correct dimensions)
            if model_name.startswith("CNN1"): # CNN1 expects [B, 1]
                targets = targets.view(-1, 1).float() # Robust reshape
            elif model_name.startswith("CNN2"): # CNN2 expects [B, 2]
                targets = targets.view(-1, 2).float() # Robust reshape

            optimizer.zero_grad(set_to_none=True) # Slightly more efficient zeroing

            # Optional: Use Mixed Precision (requires torch >= 1.6)
            # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            #     outputs = model(inputs)
            #     loss = criterion(outputs, targets)
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # Standard precision training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            if not is_tuning_run:
                train_loop.set_postfix(loss=f"{loss.item():.5f}")

        epoch_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(epoch_train_loss)
        if not is_tuning_run: train_loop.close()

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        running_val_mae = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False, ncols=80)

        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                if model_name.startswith("CNN1"): targets = targets.view(-1, 1).float()
                elif model_name.startswith("CNN2"): targets = targets.view(-1, 2).float()

                # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                mae = torch.abs(outputs - targets).mean()
                running_val_mae += mae.item()
                if not is_tuning_run:
                     val_loop.set_postfix(loss=f"{loss.item():.5f}", mae=f"{mae.item():.4f}")

        epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_mae = running_val_mae / len(val_loader) if len(val_loader) > 0 else float('inf')
        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(epoch_val_mae)
        if not is_tuning_run: val_loop.close()

        epoch_duration = time.time() - epoch_start_time

        # --- Log Epoch Results ---
        log_msg = f"Epoch {epoch+1}/{epochs} ({model_name}) | " \
                  f"Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | " \
                  f"Val MAE: {epoch_val_mae:.4f} | Time: {epoch_duration:.2f}s"

        if not is_tuning_run: print(log_msg, flush=True)
        elif (epoch + 1) == epochs: # Print only final epoch for tuning
             print(f"{log_msg} (Tuning Run Final)", flush=True)

        # --- Early Stopping & Best Model Saving ---
        if epoch_val_loss < best_val_loss - min_improvement:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            if best_model_save_path:
                try:
                    torch.save(model.state_dict(), best_model_save_path)
                    # if not is_tuning_run: print(f"    -> Val loss improved. Best model saved.", flush=True)
                except Exception as e: print(f"    -> Error saving best model: {e}", flush=True)
        elif early_stopping_patience > 0: # Only increment if ES is enabled
            epochs_no_improve += 1
            # if not is_tuning_run: print(f"    -> Val loss did not improve ({epochs_no_improve}/{early_stopping_patience}).", flush=True)

        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.6f} at epoch {best_epoch}.", flush=True)
            if best_model_save_path and os.path.exists(best_model_save_path):
                try:
                    # Load the best weights back into the model before returning
                    model.load_state_dict(torch.load(best_model_save_path, map_location=device))
                    print(f"Loaded best model weights from epoch {best_epoch}.", flush=True)
                except Exception as e: print(f"Error loading best model weights: {e}", flush=True)
            break # Exit training loop

    total_time = time.time() - start_time
    print(f"--- Training ({model_name}) finished in {total_time:.2f} seconds ---", flush=True)
    if best_epoch > 0: print(f"Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.6f}", flush=True)

    # Save history CSV
    if results_save_path and not is_tuning_run:
        try:
            pd.DataFrame(history).to_csv(results_save_path, index_label='epoch')
            print(f"Training history saved to: {results_save_path}", flush=True)
        except Exception as e: print(f"Error saving training history: {e}", flush=True)

    if is_tuning_run:
        # Return the best validation loss achieved during this tuning run
        return best_val_loss # Return best, not necessarily final
    else:
        return history


def evaluate_model(model, model_name, test_loader, criterion, device):
    """Evaluates the model on the test set."""
    print(f"\n--- Evaluating Model ({model_name}) on Test Set ---", flush=True)
    model.eval()
    running_test_loss = 0.0
    running_test_mae = 0.0
    non_blocking = device.type == 'cuda'

    if not test_loader or len(test_loader) == 0:
        print("Test loader is empty. Cannot evaluate.")
        return {'test_loss': float('nan'), 'test_mae': float('nan')}

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f"Testing ({model_name})", ncols=80):
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)

            if model_name.startswith("CNN1"): targets = targets.view(-1, 1).float()
            elif model_name.startswith("CNN2"): targets = targets.view(-1, 2).float()

            # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item()

            mae = torch.abs(outputs - targets).mean()
            running_test_mae += mae.item()

    test_loss = running_test_loss / len(test_loader)
    test_mae = running_test_mae / len(test_loader)

    print(f"Test Results ({model_name}):")
    print(f"  Test Loss (MSE): {test_loss:.6f}")
    print(f"  Test MAE:        {test_mae:.4f}")
    print("-" * 30, flush=True)

    return {'test_loss': test_loss, 'test_mae': test_mae}


def train_joint_model(cnn1_model, cnn2_model, model_name,
                      train_loader, val_loader, # val_loader uses BallLandingDataset
                      optimizer, device, epochs,
                      # Removed fixed penalty_weight from args, use config
                      R1, R2, # Pass optimized integers R1, R2
                      early_stopping_patience=10, min_improvement=1e-5,
                      results_save_path=None, best_model_save_path_cnn1=None, best_model_save_path_cnn2=None):
    """
    Trains CNN1 and CNN2 jointly with potentially adaptive penalty weight.
    CNN1 predicts h(x) scores per frame in a long sequence. Loss_CNN1 = MSE(scores, h(x)_targets).
    CNN2 predicts landing based on R1+R2+1 frames dynamically selected around CNN1's pred hit. Loss_CNN2 = MSE(coords, target_coords).
    Penalty loss based on distance between CNN1 predicted index and true index.
    Total Loss = Loss_CNN1 + Loss_CNN2 + current_penalty_weight * Loss_Penalty.
    """
    start_time = time.time()
    # History tracks combined loss and potentially individual components + adaptive weight
    history = {'train_loss': [], 'train_loss_cnn1': [], 'train_loss_cnn2': [], 'train_loss_penalty': [],
               'adaptive_penalty_weight': []}
    if val_loader:
        history['val_loss_cnn2'] = []
        history['val_mae_cnn2'] = []
    best_val_loss = float('inf') # Based on CNN2 validation loss
    epochs_no_improve = 0
    best_epoch = 0

    # Loss functions
    criterion_cnn1 = nn.MSELoss()
    criterion_cnn2 = nn.MSELoss()
    criterion_penalty = nn.L1Loss(reduction='mean') # L1 is mean abs error

    # Initialize EMA variables for smoothing losses if adaptive penalty is enabled
    ema_task_loss = None
    ema_penalty_loss = None

    # Ensure R1, R2 are positive integers
    R1 = max(1, int(round(R1)))
    R2 = max(1, int(round(R2)))
    cnn2_seq_len_dynamic = R1 + R2 + 1
    cnn2_input_channels_dynamic = cnn2_seq_len_dynamic * 3
    print(f"Joint Training using R1={R1}, R2={R2} => CNN2 Seq Len={cnn2_seq_len_dynamic}")

    # Ensure save directories exist
    if best_model_save_path_cnn1: os.makedirs(os.path.dirname(best_model_save_path_cnn1), exist_ok=True)
    if best_model_save_path_cnn2: os.makedirs(os.path.dirname(best_model_save_path_cnn2), exist_ok=True)
    if results_save_path: os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    print(f"--- Starting Joint Training ({model_name}) for {epochs} epochs ---", flush=True)
    # Check if validation is active and print early stopping info
    if val_loader:
        print(f"Early stopping (on Val CNN2 Loss): Patience={early_stopping_patience}, Min Improve={min_improvement}")
    else:
        print("INFO: val_loader is None. Skipping validation loop and early stopping.")
        early_stopping_patience = 0 # Ensure ES is off

    # Print penalty weight setting
    if config.ADAPTIVE_PENALTY_ENABLED:
        print(f"Adaptive Penalty Enabled: Base Weight={config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT}, Beta={config.ADAPTIVE_PENALTY_BETA}, Max={config.MAX_ADAPTIVE_PENALTY_WEIGHT}")
    else:
        print(f"Using Fixed Penalty Weight: {config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT}")


    for epoch in range(epochs):
        epoch_start_time = time.time()

        # --- Training Phase ---
        cnn1_model.train()
        cnn2_model.train()
        running_loss, running_loss_c1, running_loss_c2, running_loss_p = 0.0, 0.0, 0.0, 0.0
        running_adaptive_weight = 0.0 # Track average adaptive weight per epoch
        non_blocking = device.type == 'cuda'
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Joint Train]", leave=False, ncols=120) # Wider bar

        for i, (long_seq_tensors, target_weights, target_coords, targets_true_hit_idx) in enumerate(train_loop):
            # long_seq_tensors: [B, LongSeqLen, C, H, W]
            # target_weights: [B, LongSeqLen] (h(x) values for loss)
            # target_coords: [B, 2]
            # targets_true_hit_idx: [B] (index in long seq)

            batch_size = long_seq_tensors.size(0)
            long_seq_len = long_seq_tensors.size(1) # e.g., JOINT_DATASET_CONTEXT_FRAMES
            C, H, W = long_seq_tensors.shape[2], long_seq_tensors.shape[3], long_seq_tensors.shape[4]

            # Move data to device
            long_seq_tensors = long_seq_tensors.to(device, non_blocking=non_blocking)
            target_weights = target_weights.to(device, non_blocking=non_blocking).view(batch_size, long_seq_len).float()
            target_coords = target_coords.to(device, non_blocking=non_blocking).view(-1, 2).float()
            targets_true_hit_idx = targets_true_hit_idx.to(device, non_blocking=non_blocking).float() # Need float for L1 loss calc

            optimizer.zero_grad(set_to_none=True)

            # --- CNN1 Pass & Loss ---
            # Reshape: [B, LongSeqLen, C, H, W] -> [B*LongSeqLen, C, H, W]
            cnn1_inputs_flat = long_seq_tensors.view(batch_size * long_seq_len, C, H, W)
            cnn1_scores_flat = cnn1_model(cnn1_inputs_flat) # Output shape [B*LongSeqLen, 1]
            # Calculate CNN1 Loss against h(x) targets
            # Reshape target_weights: [B, LongSeqLen] -> [B*LongSeqLen, 1]
            target_weights_flat = target_weights.view(batch_size * long_seq_len, 1)
            loss_cnn1 = criterion_cnn1(cnn1_scores_flat, target_weights_flat)

            # --- Frame Selection ---
            # Reshape scores back: [B*LongSeqLen, 1] -> [B, LongSeqLen]
            cnn1_scores = cnn1_scores_flat.view(batch_size, long_seq_len)
            pred_hit_indices_long = torch.argmax(cnn1_scores, dim=1).long() # Index within the long sequence [B]

            # --- Dynamic CNN2 Input Construction ---
            cnn2_input_batch_list = []
            for b in range(batch_size):
                pred_idx = pred_hit_indices_long[b].item() # Predicted index for this sample
                # Define start/end indices relative to predicted index
                start_idx = pred_idx - R2
                end_idx = pred_idx + R1 # Inclusive end
                # Extract indices, clamping to the bounds of the long sequence
                indices = torch.arange(start_idx, end_idx + 1, device=device) # Use device for arange
                indices_clamped = torch.clamp(indices, 0, long_seq_len - 1)
                # Gather frames using clamped indices from the long sequence
                dynamic_seq_frames = long_seq_tensors[b][indices_clamped] # Shape [DynamicSeqLen, C, H, W]
                # Reshape for CNN2: [DynamicSeqLen, C, H, W] -> [C*DynamicSeqLen, H, W]
                dynamic_seq_cnn2 = dynamic_seq_frames.permute(1, 0, 2, 3).reshape(C * cnn2_seq_len_dynamic, H, W)
                cnn2_input_batch_list.append(dynamic_seq_cnn2)
            # Stack the individual CNN2 inputs into a batch
            cnn2_inputs_dynamic = torch.stack(cnn2_input_batch_list, dim=0) # [B, C*DynamicSeqLen, H, W]

            # --- CNN2 Pass & Loss ---
            pred_coords = cnn2_model(cnn2_inputs_dynamic) # Output shape [B, 2]
            loss_cnn2 = criterion_cnn2(pred_coords, target_coords)

            # --- Penalty Loss ---
            loss_penalty = criterion_penalty(pred_hit_indices_long.float(), targets_true_hit_idx)

            # --- Adaptive Penalty Weight Calculation ---
            current_penalty_weight = config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT # Default fixed weight
            if config.ADAPTIVE_PENALTY_ENABLED:
                # Detach losses before using .item() to prevent holding graph
                current_task_loss = loss_cnn1.item() + loss_cnn2.item()
                current_penalty_loss_val = loss_penalty.item()

                # Update EMAs
                if ema_task_loss is None: # First batch
                    ema_task_loss = current_task_loss
                    ema_penalty_loss = current_penalty_loss_val
                else:
                    beta = config.ADAPTIVE_PENALTY_BETA
                    ema_task_loss = beta * ema_task_loss + (1 - beta) * current_task_loss
                    ema_penalty_loss = beta * ema_penalty_loss + (1 - beta) * current_penalty_loss_val

                # Calculate adaptive weight (avoid division by zero or near-zero)
                if ema_penalty_loss > config.ADAPTIVE_PENALTY_EPSILON:
                    adaptive_weight = config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT * (ema_task_loss / ema_penalty_loss)
                else: # If penalty EMA is tiny, use a fallback
                    adaptive_weight = config.MAX_ADAPTIVE_PENALTY_WEIGHT # Avoid huge weight if task loss isn't also tiny

                # Clamp the adaptive weight
                current_penalty_weight = max(0.0, min(adaptive_weight, config.MAX_ADAPTIVE_PENALTY_WEIGHT))

            running_adaptive_weight += current_penalty_weight # Track for epoch average

            # --- Total Loss ---
            total_loss = loss_cnn1 + loss_cnn2 + current_penalty_weight * loss_penalty # Use adaptive or fixed weight

            # --- Backward and Optimize ---
            # If using gradient accumulation, divide loss before backward
            # total_loss = total_loss / accumulation_steps
            total_loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(cnn1_model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(cnn2_model.parameters(), max_norm=1.0)

            # Step optimizer (potentially handle gradient accumulation)
            # if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True) # Zero after step

            # --- Logging ---
            running_loss += total_loss.item() # Log the actual total loss used
            running_loss_c1 += loss_cnn1.item()
            running_loss_c2 += loss_cnn2.item()
            running_loss_p += loss_penalty.item() # Log unweighted penalty loss
            train_loop.set_postfix(loss=f"{total_loss.item():.4f}", pena_wt=f"{current_penalty_weight:.3f}", pen=f"{loss_penalty.item():.2f}")


        # --- End of Training Epoch ---
        epoch_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_train_loss_c1 = running_loss_c1 / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_train_loss_c2 = running_loss_c2 / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_train_loss_p = running_loss_p / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_avg_adaptive_weight = running_adaptive_weight / len(train_loader) if len(train_loader) > 0 else config.DEFAULT_JOINT_TRAINING_PENALTY_WEIGHT

        history['train_loss'].append(epoch_train_loss)
        history['train_loss_cnn1'].append(epoch_train_loss_c1)
        history['train_loss_cnn2'].append(epoch_train_loss_c2)
        history['train_loss_penalty'].append(epoch_train_loss_p)
        history['adaptive_penalty_weight'].append(epoch_avg_adaptive_weight) # Store epoch average
        train_loop.close()

        # --- Validation Phase (Optional) ---
        epoch_val_loss_c2 = float('nan') # Default if skipped
        epoch_val_mae_c2 = float('nan')
        if val_loader: # Check if validation should run
            cnn1_model.eval(); cnn2_model.eval() # Set both to eval
            running_val_loss_c2 = 0.0
            running_val_mae_c2 = 0.0
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Joint Val CNN2]", leave=False, ncols=100)
            with torch.no_grad():
                try:
                    # val_loader yields data from BallLandingDataset with dynamic length
                    for inputs_cnn2_val, targets_coords_val in val_loop:
                        inputs_cnn2_val = inputs_cnn2_val.to(device, non_blocking=non_blocking) # Shape [B, C*DynamicSeqLen, H, W]
                        targets_coords_val = targets_coords_val.to(device, non_blocking=non_blocking).view(-1, 2).float()

                        # This should now match model's expected input channels
                        outputs_coords_val = cnn2_model(inputs_cnn2_val)

                        loss_c2_val = criterion_cnn2(outputs_coords_val, targets_coords_val)
                        running_val_loss_c2 += loss_c2_val.item()
                        mae_c2_val = torch.abs(outputs_coords_val - targets_coords_val).mean()
                        running_val_mae_c2 += mae_c2_val.item()
                        val_loop.set_postfix(loss=f"{loss_c2_val.item():.5f}", mae=f"{mae_c2_val.item():.4f}")

                except Exception as e: # Catch unexpected errors during validation loop
                    print(f"\n!!! UNEXPECTED VALIDATION ERROR: {e} !!!")
                    running_val_loss_c2 = float('inf') # Mark validation as failed for this epoch
                    running_val_mae_c2 = float('inf')

            # Calculate metrics only if loop didn't break with error immediately
            if len(val_loader) > 0 and math.isfinite(running_val_loss_c2):
                 epoch_val_loss_c2 = running_val_loss_c2 / len(val_loader)
                 epoch_val_mae_c2 = running_val_mae_c2 / len(val_loader)
            else: # Handle cases where validation failed or loader was empty
                 epoch_val_loss_c2 = float('inf')
                 epoch_val_mae_c2 = float('inf')

            # Store validation results in history
            history['val_loss_cnn2'].append(epoch_val_loss_c2)
            history['val_mae_cnn2'].append(epoch_val_mae_c2)
            if 'val_loop' in locals() and val_loop is not None: # Close tqdm loop only if it was created
                val_loop.close()
        # --- End Optional Validation ---


        epoch_duration = time.time() - epoch_start_time

        # --- Log Epoch Results ---
        val_log = f"Val CNN2: Loss={epoch_val_loss_c2:.4f}, MAE={epoch_val_mae_c2:.4f}" if val_loader and math.isfinite(epoch_val_loss_c2) else "Val Skipped/Failed"
        print(f"Epoch {epoch+1}/{epochs} ({model_name}) | "
              f"Train Losses: Tot={epoch_train_loss:.4f} (C1={epoch_train_loss_c1:.4f}, C2={epoch_train_loss_c2:.4f}, Pen={epoch_train_loss_p:.2f}) | "
              f"Avg Pena Wt: {epoch_avg_adaptive_weight:.3f} | " # Log average weight used
              f"{val_log} | Time: {epoch_duration:.2f}s", flush=True)

        # --- Early Stopping Check (only if validation ran and was successful) ---
        if val_loader and math.isfinite(epoch_val_loss_c2): # Check if validation loss is valid
            current_val_loss_for_es = epoch_val_loss_c2
            if current_val_loss_for_es < best_val_loss - min_improvement:
                best_val_loss = current_val_loss_for_es
                epochs_no_improve = 0
                best_epoch = epoch + 1
                # Save BOTH models when validation improves
                if best_model_save_path_cnn1:
                    try: torch.save(cnn1_model.state_dict(), best_model_save_path_cnn1)
                    except Exception as e: print(f" Error saving best CNN1: {e}")
                if best_model_save_path_cnn2:
                    try: torch.save(cnn2_model.state_dict(), best_model_save_path_cnn2)
                    except Exception as e: print(f" Error saving best CNN2: {e}")
            elif early_stopping_patience > 0:
                epochs_no_improve += 1

            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss (CNN2): {best_val_loss:.6f} at epoch {best_epoch}.", flush=True)
                # Load best weights back
                if best_model_save_path_cnn1 and os.path.exists(best_model_save_path_cnn1):
                    try: cnn1_model.load_state_dict(torch.load(best_model_save_path_cnn1, map_location=device))
                    except Exception as e: print(f" Error loading best CNN1 wts: {e}")
                if best_model_save_path_cnn2 and os.path.exists(best_model_save_path_cnn2):
                    try: cnn2_model.load_state_dict(torch.load(best_model_save_path_cnn2, map_location=device))
                    except Exception as e: print(f" Error loading best CNN2 wts: {e}")
                if os.path.exists(best_model_save_path_cnn1) or os.path.exists(best_model_save_path_cnn2):
                    print(f"Loaded best model weights from epoch {best_epoch}.", flush=True)
                break # Exit training loop

        # --- Save model at end if validation/ES is skipped ---
        if not val_loader and (epoch + 1) == epochs:
             print("Saving models at the end of training (no validation).")
             if best_model_save_path_cnn1:
                 try: torch.save(cnn1_model.state_dict(), best_model_save_path_cnn1)
                 except Exception as e: print(f" Error saving final CNN1: {e}")
             if best_model_save_path_cnn2:
                 try: torch.save(cnn2_model.state_dict(), best_model_save_path_cnn2)
                 except Exception as e: print(f" Error saving final CNN2: {e}")

    # --- End of Training Loop ---
    total_time = time.time() - start_time
    print(f"--- Joint Training ({model_name}) finished in {total_time:.2f} seconds ---", flush=True)
    if best_epoch > 0: print(f"Best epoch (based on CNN2 Val Loss): {best_epoch}, Best Val Loss: {best_val_loss:.6f}", flush=True)
    else: print("Completed all epochs or validation was skipped.")

    # Save history CSV
    if results_save_path:
        try:
            # Ensure all history lists have the same length before creating DataFrame
            max_len = len(history['train_loss'])
            for key in history:
                 if len(history[key]) < max_len:
                      history[key].extend([float('nan')] * (max_len - len(history[key]))) # Pad with NaN if needed

            pd.DataFrame(history).to_csv(results_save_path, index_label='epoch')
            print(f"Joint history (incl. adaptive weight) saved to: {results_save_path}", flush=True)
        except Exception as e: print(f"Error saving joint training history: {e}", flush=True)

    return history