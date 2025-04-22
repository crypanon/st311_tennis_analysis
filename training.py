# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import pandas as pd
from tqdm import tqdm # Use standard tqdm

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


# --- MODIFIED Joint Training Function ---
def train_joint_model(cnn1_model, cnn2_model, model_name,
                      train_loader, val_loader, # val_loader uses BallLandingDataset
                      optimizer, device, epochs, penalty_weight,
                      R1, R2, # Pass optimized integers R1, R2
                      early_stopping_patience=10, min_improvement=1e-5,
                      results_save_path=None, best_model_save_path_cnn1=None, best_model_save_path_cnn2=None):
    """
    Trains CNN1 and CNN2 jointly.
    CNN1 predicts h(x) scores per frame in a long sequence. Loss_CNN1 = MSE(scores, h(x)_targets).
    CNN2 predicts landing based on R1+R2+1 frames dynamically selected around CNN1's pred hit. Loss_CNN2 = MSE(coords, target_coords).
    Penalty loss based on distance between CNN1 predicted index and true index.
    """
    start_time = time.time()
    history = {'train_loss': [], 'train_loss_cnn1': [], 'train_loss_cnn2': [], 'train_loss_penalty': [],
               'val_loss_cnn2': [], 'val_mae_cnn2': []} # More detailed history
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0

    criterion_cnn1 = nn.MSELoss() # Match CNN1 scores against h(x) targets
    criterion_cnn2 = nn.MSELoss()
    criterion_penalty = nn.L1Loss(reduction='mean')

    # Ensure R1, R2 are positive integers
    R1 = max(1, int(round(R1)))
    R2 = max(1, int(round(R2)))
    cnn2_seq_len_dynamic = R1 + R2 + 1
    cnn2_input_channels_dynamic = cnn2_seq_len_dynamic * 3
    print(f"Joint Training using R1={R1}, R2={R2} => CNN2 Seq Len={cnn2_seq_len_dynamic}")

    # Check compatibility between dynamic CNN2 input and loaded CNN2 model (if pre-trained)
    # This check assumes cnn2_model is already instantiated correctly outside this func
    # We might need to instantiate CNN2 *inside* if R1/R2 change? No, R1/R2 are fixed post-opt.


    if best_model_save_path_cnn1: os.makedirs(os.path.dirname(best_model_save_path_cnn1), exist_ok=True)
    if best_model_save_path_cnn2: os.makedirs(os.path.dirname(best_model_save_path_cnn2), exist_ok=True)
    if results_save_path: os.makedirs(os.path.dirname(results_save_path), exist_ok=True)

    print(f"--- Starting Joint Training ({model_name}) for {epochs} epochs ---", flush=True)
    print(f"Penalty Weight: {penalty_weight}")
    # ... (early stopping print) ...

    for epoch in range(epochs):
        epoch_start_time = time.time()
        cnn1_model.train(); cnn2_model.train()
        running_loss, running_loss_c1, running_loss_c2, running_loss_p = 0.0, 0.0, 0.0, 0.0
        non_blocking = device.type == 'cuda'
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Joint Train]", leave=False, ncols=120) # Wider bar

        for long_seq_tensors, target_weights, target_coords, targets_true_hit_idx in train_loop:
            # long_seq_tensors: [B, LongSeqLen, C, H, W]
            # target_weights: [B, LongSeqLen] (h(x) values for loss)
            # target_coords: [B, 2]
            # targets_true_hit_idx: [B] (index in long seq)

            batch_size = long_seq_tensors.size(0)
            long_seq_len = long_seq_tensors.size(1) # e.g., JOINT_DATASET_CONTEXT_FRAMES

            long_seq_tensors = long_seq_tensors.to(device, non_blocking=non_blocking)
            target_weights = target_weights.to(device, non_blocking=non_blocking).view(batch_size, long_seq_len).float()
            target_coords = target_coords.to(device, non_blocking=non_blocking).view(-1, 2).float()
            targets_true_hit_idx = targets_true_hit_idx.to(device, non_blocking=non_blocking).float()

            optimizer.zero_grad(set_to_none=True)

            # --- CNN1 Pass ---
            # Reshape: [B, LongSeqLen, C, H, W] -> [B*LongSeqLen, C, H, W]
            cnn1_inputs_flat = long_seq_tensors.view(batch_size * long_seq_len, *long_seq_tensors.shape[2:])
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
            # Create batch of sequences for CNN2 input [B, C*DynamicSeqLen, H, W]
            # This is the most complex part
            cnn2_input_batch_list = []
            C, H, W = long_seq_tensors.shape[2], long_seq_tensors.shape[3], long_seq_tensors.shape[4]
            for b in range(batch_size):
                pred_idx = pred_hit_indices_long[b].item() # Predicted index for this sample
                # Define start/end indices relative to predicted index
                start_idx = pred_idx - R2
                end_idx = pred_idx + R1 # Inclusive end

                # Extract indices, clamping to the bounds of the long sequence
                indices = torch.arange(start_idx, end_idx + 1, device=device)
                indices_clamped = torch.clamp(indices, 0, long_seq_len - 1)

                # Gather frames using clamped indices from the long sequence
                # long_seq_tensors[b] shape is [LongSeqLen, C, H, W]
                # Need indices_clamped to index dim 0
                dynamic_seq_frames = long_seq_tensors[b][indices_clamped] # Shape [DynamicSeqLen, C, H, W]

                # Reshape for CNN2: [DynamicSeqLen, C, H, W] -> [C*DynamicSeqLen, H, W]
                dynamic_seq_cnn2 = dynamic_seq_frames.permute(1, 0, 2, 3).reshape(C * cnn2_seq_len_dynamic, H, W)
                cnn2_input_batch_list.append(dynamic_seq_cnn2)

            # Stack the individual CNN2 inputs into a batch
            cnn2_inputs_dynamic = torch.stack(cnn2_input_batch_list, dim=0) # [B, C*DynamicSeqLen, H, W]

            # --- CNN2 Pass ---
            pred_coords = cnn2_model(cnn2_inputs_dynamic) # Output shape [B, 2]
            loss_cnn2 = criterion_cnn2(pred_coords, target_coords)

            # --- Penalty Loss ---
            loss_penalty = criterion_penalty(pred_hit_indices_long.float(), targets_true_hit_idx)

            # --- Total Loss ---
            total_loss = loss_cnn1 + loss_cnn2 + penalty_weight * loss_penalty

            # --- Backward and Optimize ---
            total_loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(cnn1_model.parameters(), max_norm=1.0)
            # torch.nn.utils.clip_grad_norm_(cnn2_model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += total_loss.item()
            running_loss_c1 += loss_cnn1.item()
            running_loss_c2 += loss_cnn2.item()
            running_loss_p += loss_penalty.item()

            train_loop.set_postfix(loss=f"{total_loss.item():.4f}", c1=f"{loss_cnn1.item():.4f}", c2=f"{loss_cnn2.item():.4f}", pen=f"{loss_penalty.item():.2f}")

        # --- End of Epoch ---
        epoch_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_train_loss_c1 = running_loss_c1 / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_train_loss_c2 = running_loss_c2 / len(train_loader) if len(train_loader) > 0 else 0.0
        epoch_train_loss_p = running_loss_p / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(epoch_train_loss)
        history['train_loss_cnn1'].append(epoch_train_loss_c1)
        history['train_loss_cnn2'].append(epoch_train_loss_c2)
        history['train_loss_penalty'].append(epoch_train_loss_p)
        train_loop.close()

        # --- Validation Phase (Evaluate CNN2 on standard BallLandingDataset) ---
        # ... (Validation logic remains the same as before, using cnn2_model.eval()) ...
        # ... It appends to history['val_loss_cnn2'] and history['val_mae_cnn2'] ...
        # --- Validation Phase ---
        cnn1_model.eval(); cnn2_model.eval() # Set both to eval
        running_val_loss_c2 = 0.0
        running_val_mae_c2 = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Joint Val CNN2]", leave=False, ncols=100)
        with torch.no_grad():
            for inputs_cnn2_val, targets_coords_val in val_loop: # Uses BallLandingDataset loader
                inputs_cnn2_val = inputs_cnn2_val.to(device, non_blocking=non_blocking)
                targets_coords_val = targets_coords_val.to(device, non_blocking=non_blocking).view(-1, 2).float()
                outputs_coords_val = cnn2_model(inputs_cnn2_val)
                loss_c2_val = criterion_cnn2(outputs_coords_val, targets_coords_val)
                running_val_loss_c2 += loss_c2_val.item()
                mae_c2_val = torch.abs(outputs_coords_val - targets_coords_val).mean()
                running_val_mae_c2 += mae_c2_val.item()
                val_loop.set_postfix(loss=f"{loss_c2_val.item():.5f}", mae=f"{mae_c2_val.item():.4f}")
        epoch_val_loss_c2 = running_val_loss_c2 / len(val_loader) if len(val_loader) > 0 else float('inf')
        epoch_val_mae_c2 = running_val_mae_c2 / len(val_loader) if len(val_loader) > 0 else float('inf')
        history['val_loss_cnn2'].append(epoch_val_loss_c2)
        history['val_mae_cnn2'].append(epoch_val_mae_c2)
        val_loop.close()
        epoch_duration = time.time() - epoch_start_time

        # --- Log Epoch Results (more detailed) ---
        print(f"Epoch {epoch+1}/{epochs} ({model_name}) | "
              f"Train Losses: Tot={epoch_train_loss:.4f} (C1={epoch_train_loss_c1:.4f}, C2={epoch_train_loss_c2:.4f}, Pen={epoch_train_loss_p:.2f}) | "
              f"Val CNN2: Loss={epoch_val_loss_c2:.4f}, MAE={epoch_val_mae_c2:.4f} | "
              f"Time: {epoch_duration:.2f}s", flush=True)

        # --- Early Stopping (based on CNN2 validation loss) ---
        # ... (Early stopping logic remains the same, saving both models) ...
        current_val_loss_for_es = epoch_val_loss_c2
        if current_val_loss_for_es < best_val_loss - min_improvement:
            best_val_loss = current_val_loss_for_es
            epochs_no_improve = 0
            best_epoch = epoch + 1
            if best_model_save_path_cnn1: torch.save(cnn1_model.state_dict(), best_model_save_path_cnn1)
            if best_model_save_path_cnn2: torch.save(cnn2_model.state_dict(), best_model_save_path_cnn2)
        elif early_stopping_patience > 0:
            epochs_no_improve += 1
        if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss (CNN2): {best_val_loss:.6f} at epoch {best_epoch}.", flush=True)
            if best_model_save_path_cnn1 and os.path.exists(best_model_save_path_cnn1): cnn1_model.load_state_dict(torch.load(best_model_save_path_cnn1, map_location=device))
            if best_model_save_path_cnn2 and os.path.exists(best_model_save_path_cnn2): cnn2_model.load_state_dict(torch.load(best_model_save_path_cnn2, map_location=device))
            print(f"Loaded best model weights from epoch {best_epoch}.", flush=True)
            break

    # --- End of Training ---
    # ... (Final print and history saving remain the same) ...
    total_time = time.time() - start_time
    print(f"--- Joint Training ({model_name}) finished in {total_time:.2f} seconds ---", flush=True)
    if best_epoch > 0: print(f"Best epoch (based on CNN2 Val Loss): {best_epoch}, Best Val Loss: {best_val_loss:.6f}", flush=True)
    if results_save_path:
        try: pd.DataFrame(history).to_csv(results_save_path, index_label='epoch'); print(f"Joint history saved to: {results_save_path}")
        except Exception as e: print(f"Error saving joint training history: {e}")
    return history