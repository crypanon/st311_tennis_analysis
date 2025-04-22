# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image

# Assuming augmentations.py is in the same directory or accessible
from augmentations import apply_augmentations
from config import IMG_HEIGHT, IMG_WIDTH, JOINT_DATASET_CONTEXT_FRAMES # Use context length

class TennisFrameDataset(Dataset):
    """Dataset for CNN1 (Hit Frame Regression). Loads single frames."""
    def __init__(self, paths, targets, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, augment=False):
        self.paths = paths
        self.targets = targets # These are weights for CNN1
        self.img_height = img_height
        self.img_width = img_width
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        target = self.targets[idx]

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: raise IOError(f"Failed to load image: {img_path}")

            if self.augment:
                img_pil_augmented, _ = apply_augmentations(img_bgr, cnn_type=1)
                img_rgb = np.array(img_pil_augmented) # PIL (RGB) -> NumPy (RGB)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # OpenCV (BGR) -> NumPy (RGB)

            # Common processing (Resize, Normalize, Transpose)
            img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
            # Normalize to [0, 1] and change to float32 *before* transpose
            img_normalized = (img_resized / 255.0).astype(np.float32)
            # Transpose HWC to CHW
            img_transposed = np.transpose(img_normalized, (2, 0, 1))

            # Convert to PyTorch tensors
            # Using torch.from_numpy is often more efficient than torch.tensor
            img_tensor = torch.from_numpy(img_transposed)
            target_tensor = torch.tensor(target, dtype=torch.float32)

            return img_tensor, target_tensor

        except Exception as e:
            print(f"Error processing image {img_path} at index {idx}: {e}. Returning zeros.")
            # Return tensors with correct shape and type
            dummy_img = torch.zeros((3, self.img_height, self.img_width), dtype=torch.float32)
            dummy_target = torch.tensor(0.0, dtype=torch.float32)
            return dummy_img, dummy_target


class BallLandingDataset(Dataset):
    """Dataset for CNN2 (Landing Spot). Loads frame sequences."""
    def __init__(self, sequence_data, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, n_frames_sequence=7, augment=False):
        self.sequence_data = sequence_data
        self.img_height = img_height
        self.img_width = img_width
        self.n_frames_sequence = n_frames_sequence
        self.augment = augment

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        item = self.sequence_data[idx]
        frame_paths = item['sequence_paths']
        # Make target coords mutable for potential flipping during augmentation
        target_coords = list(item['target_coords'])

        sequence_tensors = []
        global_flip_status = False # Track if flip augmentation was applied

        for i, frame_path in enumerate(frame_paths):
            try:
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None: raise IOError(f"Failed to load frame: {frame_path}")

                if self.augment:
                    # Apply augmentations; use flip status from the first frame for consistency?
                    # Current approach: Check flip status for the first frame only
                    img_pil_augmented, was_flipped = apply_augmentations(img_bgr, cnn_type=2)
                    img_rgb = np.array(img_pil_augmented)
                    if i == 0: # Use first frame's flip status for coordinate adjustment
                         global_flip_status = was_flipped
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Common processing
                img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
                img_normalized = (img_resized / 255.0).astype(np.float32)
                img_transposed = np.transpose(img_normalized, (2, 0, 1)) # CHW
                sequence_tensors.append(torch.from_numpy(img_transposed))

            except Exception as e:
                print(f"Error processing frame {frame_path} in sequence {idx}: {e}. Using black frame.")
                black_frame = torch.zeros((3, self.img_height, self.img_width), dtype=torch.float32)
                sequence_tensors.append(black_frame)
                if i == 0: global_flip_status = False # Ensure status is set

        # Post-Augmentation Coordinate Adjustment
        if global_flip_status:
            target_coords[0] = 1.0 - target_coords[0] # Flip X coordinate

        # Stack tensors along a new dimension (Time/Sequence) before concatenating channels
        # Result shape: [SeqLen, Channels, Height, Width]
        try:
            # Use torch.stack for creating the sequence dimension
            stacked_sequence = torch.stack(sequence_tensors, dim=0)

            # Reshape/Permute to [Channels * SeqLen, Height, Width] for Conv2D input
            # Target shape: [C*T, H, W]
            num_channels, height, width = stacked_sequence.shape[1], stacked_sequence.shape[2], stacked_sequence.shape[3]
            # Permute to [C, T, H, W] then reshape
            final_tensor = stacked_sequence.permute(1, 0, 2, 3).reshape(num_channels * self.n_frames_sequence, height, width)

            target_tensor = torch.tensor(target_coords, dtype=torch.float32)
            return final_tensor, target_tensor

        except Exception as e:
             print(f"Error stacking/reshaping sequence {idx}: {e}. Returning zeros.")
             dummy_seq = torch.zeros((self.n_frames_sequence * 3, self.img_height, self.img_width), dtype=torch.float32)
             dummy_target = torch.zeros((2,), dtype=torch.float32)
             return dummy_seq, dummy_target
        
# --- MODIFIED Dataset for Joint Training ---
class JointPredictionDataset(Dataset):
    """
    Dataset for Joint Training. Loads a LONG context window of frames centered
    on TRUE hit frame. Provides target h(x) weights, landing coords, and
    the index of the true hit frame within the long sequence.
    """
    def __init__(self, sequence_data, img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 n_frames_context=JOINT_DATASET_CONTEXT_FRAMES, augment=False):
        # sequence_data expects list of dicts from get_long_context_sequences
        self.sequence_data = sequence_data
        self.img_height = img_height
        self.img_width = img_width
        self.n_frames_context = n_frames_context # Length of the long sequence
        self.augment = augment
        if n_frames_context % 2 == 0:
             print(f"Warning: Joint dataset context length ({n_frames_context}) should be odd.")

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        item = self.sequence_data[idx]
        frame_paths = item['sequence_paths']
        target_coords = list(item['target_coords'])
        target_weights = item['target_weights'] # List of h(x) target scores
        true_hit_index = item['true_hit_index_in_sequence'] # Index within this long seq

        if len(frame_paths) != self.n_frames_context or len(target_weights) != self.n_frames_context:
            print(f"Error: Data mismatch for joint dataset item {idx}. "
                  f"Expected {self.n_frames_context} frames/weights, got {len(frame_paths)}/{len(target_weights)}. Returning zeros.")
            return self._get_zero_tensors() # Return dummy data on error

        long_sequence_frame_tensors = []
        global_flip_status = False

        # Process each frame in the LONG sequence
        for i, frame_path in enumerate(frame_paths):
            try:
                img_bgr = cv2.imread(frame_path)
                if img_bgr is None: raise IOError(f"Failed load frame: {frame_path}")

                if self.augment:
                    # Augmentations might affect CNN1 scores, apply carefully if needed
                    # Keeping CNN2 type aug for now
                    img_pil_augmented, was_flipped = apply_augmentations(img_bgr, cnn_type=2)
                    img_rgb = np.array(img_pil_augmented)
                    if i == 0: global_flip_status = was_flipped
                else:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Common processing
                img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
                img_normalized = (img_resized / 255.0).astype(np.float32)
                img_transposed = np.transpose(img_normalized, (2, 0, 1)) # CHW
                long_sequence_frame_tensors.append(torch.from_numpy(img_transposed))

            except Exception as e:
                print(f"Error processing frame {frame_path} in long joint sequence {idx}: {e}. Returning zeros.")
                return self._get_zero_tensors() # Return dummy data

        # Coordinate adjustment based on flip status
        if global_flip_status:
            target_coords[0] = 1.0 - target_coords[0]

        try:
            # Stack frames for CNN1 input: [ContextLen, C, H, W]
            long_seq_tensor = torch.stack(long_sequence_frame_tensors, dim=0)

            # Prepare target tensors
            target_coords_tensor = torch.tensor(target_coords, dtype=torch.float32)
            target_weights_tensor = torch.tensor(target_weights, dtype=torch.float32) # [ContextLen]
            true_hit_index_tensor = torch.tensor(true_hit_index, dtype=torch.long)

            # Returns:
            # 1. Long sequence tensor [ContextLen, C, H, W]
            # 2. Target h(x) weights for the long sequence [ContextLen]
            # 3. Target landing coordinates [2]
            # 4. Index of true hit frame within the long sequence [1]
            return long_seq_tensor, target_weights_tensor, target_coords_tensor, true_hit_index_tensor

        except Exception as e:
             print(f"Error stacking/tensorizing long joint sequence {idx}: {e}. Returning zeros.")
             return self._get_zero_tensors()

    def _get_zero_tensors(self):
        """Helper to return zero tensors on error."""
        dummy_seq = torch.zeros((self.n_frames_context, 3, self.img_height, self.img_width), dtype=torch.float32)
        dummy_weights = torch.zeros((self.n_frames_context,), dtype=torch.float32)
        dummy_coords = torch.zeros((2,), dtype=torch.float32)
        dummy_index = torch.tensor(0, dtype=torch.long)
        return dummy_seq, dummy_weights, dummy_coords, dummy_index