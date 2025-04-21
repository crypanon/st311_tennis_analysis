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