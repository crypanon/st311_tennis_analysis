import os
import torch
import cv2
import numpy as np
import re
from tqdm import tqdm

# Import project modules
from config import IMG_HEIGHT, IMG_WIDTH, DOUBLES_COURT_WIDTH_M, HALF_COURT_LENGTH_M
from models import HitFrameRegressorFinal, LandingPointCNN

# --- Model Loading Helpers ---
def load_final_cnn1_model(model_path, device, arch_params):
    """Loads the final trained HitFrameRegressor model."""
    if not os.path.exists(model_path):
        print(f"Error: CNN1 model file not found at {model_path}")
        return None
    try:
        model = HitFrameRegressorFinal(
            block_filters=tuple(arch_params['filters']), # Ensure tuple
            fc_size=arch_params['fc_size'],
            dropout_rate=arch_params['dropout']
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Final CNN1 model loaded successfully from {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"Error loading CNN1 model state dict: {e}")
        return None

def load_final_cnn2_model(model_path, device, input_channels):
    """Loads the final trained LandingPointCNN model."""
    if not os.path.exists(model_path):
        print(f"Error: CNN2 model file not found at {model_path}")
        return None
    try:
        model = LandingPointCNN(
            input_channels=input_channels, output_dim=2
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Final CNN2 model loaded successfully from {os.path.basename(model_path)}")
        return model
    except Exception as e:
        print(f"Error loading CNN2 model state dict: {e}")
        return None

# --- Coordinate Denormalization ---
def denormalize_coordinates(norm_x, norm_y):
    """Converts normalized [0, 1] coordinates back to approx metric distances."""
    norm_x, norm_y = np.clip(norm_x, 0, 1), np.clip(norm_y, 0, 1)
    dist_from_left_m = norm_x * DOUBLES_COURT_WIDTH_M
    dist_from_right_m = DOUBLES_COURT_WIDTH_M - dist_from_left_m
    dist_from_baseline_m = norm_y * HALF_COURT_LENGTH_M # Y=0 is baseline, Y=1 is net
    return dist_from_left_m, dist_from_right_m, dist_from_baseline_m

# --- Full Prediction Pipeline ---
def predict_hit_and_landing(cnn1_model, cnn2_model, frames_directory,
                            cnn2_seq_len, device):
    """
    Runs the prediction pipeline using pre-loaded models.

    Args:
        cnn1_model: Pre-loaded and eval-mode CNN1 model.
        cnn2_model: Pre-loaded and eval-mode CNN2 model.
        frames_directory (str): Path to the directory containing frames for one shot.
        cnn2_seq_len (int): The sequence length CNN2 expects.
        device: The torch device ('cuda' or 'cpu').

    Returns:
        tuple: (predicted_norm_coords, predicted_hit_frame_path) or (None, None) on failure.
               predicted_norm_coords is (norm_x, norm_y)
    """
    print(f"\n--- Running Prediction Pipeline for: {os.path.basename(frames_directory)} ---")
    if not cnn1_model or not cnn2_model:
         print("Error: Models not loaded.")
         return None, None

    # --- Part 1: Find Hit Frame using CNN1 ---
    try: # Robustly list, sort, and filter image files
        all_files = os.listdir(frames_directory)
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        frames_data = []
        for fname in all_files:
            if fname.lower().endswith(image_extensions):
                match = re.search(r'frame_(\d+)', fname, re.IGNORECASE)
                if match:
                    try: frames_data.append({'path': os.path.join(frames_directory, fname), 'number': int(match.group(1))})
                    except ValueError: pass # Ignore if number isn't valid int
        if not frames_data: raise ValueError("No frames with parsable numbers found.")
        sorted_frames = sorted(frames_data, key=lambda x: x['number'])
        sorted_frame_paths = [f['path'] for f in sorted_frames]
        num_total_frames = len(sorted_frame_paths)
        if num_total_frames == 0: raise ValueError("No valid image frames found.")
    except Exception as e:
        print(f"Error reading/sorting frames in {frames_directory}: {e}")
        return None, None

    # CNN1 Inference
    predictions_cnn1 = []
    with torch.no_grad():
        for frame_path in tqdm(sorted_frame_paths, desc="CNN1 Inference", leave=False, ncols=80):
            try:
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None: continue # Skip if load fails

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                img_norm = (img_resized / 255.0).astype(np.float32)
                img_chw = np.transpose(img_norm, (2, 0, 1))
                inp_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)

                # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                score = cnn1_model(inp_tensor).item()
                predictions_cnn1.append({'path': frame_path, 'score': score})
            except Exception as e:
                print(f"Warn: CNN1 error on frame {os.path.basename(frame_path)}: {e}")

    if not predictions_cnn1:
        print("Error: CNN1 failed to produce any predictions.")
        return None, None

    best_hit = max(predictions_cnn1, key=lambda x: x['score'])
    predicted_hit_frame_path = best_hit['path']
    print(f"Predicted Hit Frame: {os.path.basename(predicted_hit_frame_path)} (Score: {best_hit['score']:.4f})")

    # --- Part 2: Construct Sequence for CNN2 ---
    try:
        hit_frame_index = sorted_frame_paths.index(predicted_hit_frame_path)
    except ValueError:
        print("Error: Predicted hit frame path not found in sorted list.")
        return None, predicted_hit_frame_path # Return hit path even if landing fails

    half_window = cnn2_seq_len // 2
    start_idx = hit_frame_index - half_window
    end_idx = hit_frame_index + half_window # Inclusive index for range end
    # Pad sequence by clamping indices
    sequence_paths = [sorted_frame_paths[np.clip(i, 0, num_total_frames - 1)] for i in range(start_idx, end_idx + 1)]

    if len(sequence_paths) != cnn2_seq_len:
        print(f"Error: Final sequence length mismatch ({len(sequence_paths)} vs {cnn2_seq_len}).")
        return None, predicted_hit_frame_path

    # --- Part 3: Predict Landing using CNN2 ---
    sequence_tensors = []
    try:
        for frame_path in sequence_paths:
            img_bgr = cv2.imread(frame_path)
            if img_bgr is None: raise ValueError(f"Failed to load {os.path.basename(frame_path)} for CNN2 sequence")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
            img_norm = (img_resized / 255.0).astype(np.float32)
            img_chw = np.transpose(img_norm, (2, 0, 1))
            sequence_tensors.append(torch.from_numpy(img_chw)) # Use from_numpy

        # Stack along time, permute, and reshape
        stacked_sequence = torch.stack(sequence_tensors, dim=0) # [T, C, H, W]
        C, H, W = stacked_sequence.shape[1], stacked_sequence.shape[2], stacked_sequence.shape[3]
        # Permute to [C, T, H, W] then reshape to [C*T, H, W]
        input_tensor_cnn2 = stacked_sequence.permute(1, 0, 2, 3).reshape(C * cnn2_seq_len, H, W)
        input_batch_cnn2 = input_tensor_cnn2.unsqueeze(0).to(device) # Add batch dim

    except Exception as e:
        print(f"Error preprocessing sequence for CNN2: {e}")
        return None, predicted_hit_frame_path

    # CNN2 Inference
    with torch.no_grad():
        # with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        pred_coords_tensor = cnn2_model(input_batch_cnn2)

    pred_norm_x = pred_coords_tensor[0, 0].item()
    pred_norm_y = pred_coords_tensor[0, 1].item()
    predicted_norm_coords = (pred_norm_x, pred_norm_y)
    print(f"Predicted Landing (Normalized): ({pred_norm_x:.4f}, {pred_norm_y:.4f})")

    print("--- Pipeline Finished ---")
    return predicted_norm_coords, predicted_hit_frame_path