# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import cv2
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import constants from config
from config import DATASET_BASE_PATH, DOUBLES_COURT_WIDTH_M, HALF_COURT_LENGTH_M

# --- Metadata Loading ---
def load_metadata(csv_path, dataset_base_path):
    """Loads the main CSV, constructs absolute paths, and adds video_id."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")

    print("Loading metadata CSV...")
    df = pd.read_csv(csv_path)
    print(f"Original CSV loaded with {len(df)} rows.")

    # Construct Full Paths and Validate
    df['frame_path'] = df['frame_path'].apply(
        lambda x: os.path.join(dataset_base_path, x.replace('\\', '/'))
    )

    # Add video_id (directory containing the frame)
    df['video_id'] = df['frame_path'].apply(lambda x: os.path.dirname(x))
    print("Added 'video_id' column.")

    print(f"Hit frames (is_hit_frame == 1): {len(df[df['is_hit_frame'] == 1])}")
    print(f"Non-hit frames (is_hit_frame == 0): {len(df[df['is_hit_frame'] == 0])}")

    # Optional: Check sample path existence (can be slow for large datasets)
    # sample_path = df['frame_path'].iloc[0]
    # if not os.path.exists(sample_path):
    #     print(f"WARNING: Sample frame path does NOT exist: {sample_path}")
    # else:
    #     print("Sample frame path exists.")

    return df


# --- Frame Weight Assignment (CNN1 Target) ---
def assign_weights(group, total_frames_window, decay_rate):
    """Assigns weights based on proximity to the middle hit frame."""
    group = group.copy()
    group['weight'] = 0.0
    hit_frames = group[group['is_hit_frame'] == 1]
    if hit_frames.empty: return group

    middle_hit_idx = len(hit_frames) // 2
    middle_hit_row = hit_frames.iloc[middle_hit_idx]
    middle_hit_frame_index_in_group = middle_hit_row.name # Original DF index

    # Extract frame number robustly
    middle_frame_num = None
    try:
        match = re.search(r'frame_(\d+)', os.path.basename(middle_hit_row['frame_path']))
        if match: middle_frame_num = int(match.group(1))
    except Exception: pass

    if middle_frame_num is None: # Fallback to relative index if parsing fails
        middle_frame_num = group.index.get_loc(middle_hit_frame_index_in_group)

    window_half = total_frames_window // 2
    weights = {}
    for idx, row in group.iterrows():
        current_frame_num = None
        try:
            match = re.search(r'frame_(\d+)', os.path.basename(row['frame_path']))
            if match: current_frame_num = int(match.group(1))
        except Exception: pass
        if current_frame_num is None: current_frame_num = group.index.get_loc(idx)

        distance = abs(current_frame_num - middle_frame_num)
        if idx == middle_hit_frame_index_in_group:
            weight = 1.0
        elif distance <= window_half:
            weight = max(0.0, 1.0 - (distance * decay_rate))
        else:
            weight = 0.0
        weights[idx] = weight

    group['weight'] = group.index.map(weights)
    return group

def apply_weighting_to_df(input_df, n_frames_weighting, weight_decay):
    """Applies weight assignment to the entire DataFrame."""
    print(f"Applying weight assignment (Window: {n_frames_weighting}, Decay: {weight_decay})...")
    if 'video_id' not in input_df.columns:
         raise ValueError("Missing 'video_id' column for weighting.")

    df_weighted = input_df.copy()
    df_weighted['weight'] = 0.0 # Initialize

    # Using progress_apply for visual feedback with tqdm
    tqdm.pandas(desc="Assigning Weights")
    df_weighted = df_weighted.groupby('video_id', group_keys=False).progress_apply(
        lambda grp: assign_weights(grp, total_frames_window=n_frames_weighting, decay_rate=weight_decay)
    )
    df_weighted = df_weighted.reset_index(drop=True) # Drop old index after apply

    print("Weight assignment complete.")
    print(f"Total frames with weight > 0: {len(df_weighted[df_weighted['weight'] > 0])}")
    return df_weighted


# --- Data Balancing and Splitting ---
def balance_and_split_data(input_df, balance_ratio, test_size=0.15, val_size=0.15, random_state=42):
    """Balances based on weight > 0 and splits into train/val/test."""
    print(f"Balancing dataset (Ratio Non-Hit/Weighted: {balance_ratio}) and Splitting...")
    if 'weight' not in input_df.columns:
        raise ValueError("'weight' column missing for balancing.")

    hit_frames_weighted = input_df[input_df['weight'] > 0].copy()
    non_hit_frames = input_df[input_df['weight'] == 0].copy()

    print(f"Frames with weight > 0: {len(hit_frames_weighted)}")
    print(f"Frames with weight == 0: {len(non_hit_frames)}")

    if hit_frames_weighted.empty:
        print("Warning: No frames with weight > 0 found. Balancing skipped.")
        balanced_df = input_df.copy()
    elif non_hit_frames.empty:
        print("Warning: No non-hit frames found. Using only weighted frames.")
        balanced_df = hit_frames_weighted.copy()
    else:
        target_non_hit_count = int(balance_ratio * len(hit_frames_weighted))
        sample_size = min(len(non_hit_frames), target_non_hit_count)
        print(f"Sampling {sample_size} non-hit frames...")
        sampled_non_hit = non_hit_frames.sample(n=sample_size, random_state=random_state)
        balanced_df = pd.concat([hit_frames_weighted, sampled_non_hit])

    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"Balanced dataset size: {len(balanced_df)}")

    # Prepare for splitting
    all_paths = balanced_df['frame_path'].tolist()
    all_targets = balanced_df['weight'].tolist() # Use 'weight' for CNN1 targets
    stratify_labels = (np.array(all_targets) > 0).astype(int)

    if len(all_paths) < 3:
        print("Error: Not enough data for train/val/test split.")
        return balanced_df, [], [], [], [], [], [], []

    # Split off Test
    try:
        train_val_paths, test_paths, train_val_targets, test_targets = train_test_split(
            all_paths, all_targets, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )
    except ValueError: # Handle cases where stratification isn't possible
        print("Warning: Stratification failed for test split. Using non-stratified.")
        train_val_paths, test_paths, train_val_targets, test_targets = train_test_split(
            all_paths, all_targets, test_size=test_size, random_state=random_state
        )


    # Split Train/Val from remainder
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
    if len(train_val_paths) < 2:
        print("Warning: Not enough data left for train/validation split.")
        train_paths, val_paths = train_val_paths, []
        train_targets, val_targets = train_val_targets, []
    else:
        try:
            stratify_tv_labels = (np.array(train_val_targets) > 0).astype(int)
            train_paths, val_paths, train_targets, val_targets = train_test_split(
                train_val_paths, train_val_targets, test_size=relative_val_size, random_state=random_state, stratify=stratify_tv_labels
            )
        except ValueError:
             print("Warning: Stratification failed for train/val split. Using non-stratified.")
             train_paths, val_paths, train_targets, val_targets = train_test_split(
                 train_val_paths, train_val_targets, test_size=relative_val_size, random_state=random_state
             )

    print(f"Data Splitting Results:")
    print(f"  Training samples:   {len(train_paths)}")
    print(f"  Validation samples: {len(val_paths)}")
    print(f"  Test samples:       {len(test_paths)}")

    # Return balanced DF and split lists (paths and targets)
    return balanced_df, train_paths, train_targets, val_paths, val_targets, test_paths, test_targets


# --- Landing Coordinate Loading (CNN2 Target) ---
def map_coordinates(dist_sideline, dist_baseline, shot_type, court_width, court_length):
    """Maps raw distances (m) to normalized [0, 1] coordinates."""
    if pd.isna(dist_sideline) or pd.isna(dist_baseline) or pd.isna(shot_type) or \
       shot_type not in ['Straight', 'Cross'] or court_width <= 0 or court_length <= 0:
        return None, None # Return tuple of Nones

    y_norm = (dist_baseline / court_length)

    if shot_type == 'Straight': # Lands near RIGHT sideline (X near 1)
        x_norm = (court_width - dist_sideline) / court_width
    else: # Cross, lands near LEFT sideline (X near 0)
        x_norm = dist_sideline / court_width

    return np.clip(x_norm, 0.0, 1.0), np.clip(y_norm, 0.0, 1.0)


def load_landing_data(dataset_base_path):
    """Loads and processes landing location CSVs."""
    csv_base_path = os.path.join(dataset_base_path, 'Videos')
    print(f"Looking for landing CSVs in: {csv_base_path}")
    if not os.path.isdir(csv_base_path):
        raise FileNotFoundError(f"Landing CSV directory not found: {csv_base_path}")

    csv_files = {
        "Indoor_Cross": "Indoor Field - Crosscourt Shot.csv",
        "Indoor_Straight": "Indoor Field - Straight Shot.csv",
        "Outdoor_Cross": "Outdoor Field - Crosscourt Shot.csv",
        "Outdoor_Straight": "Outdoor Field - Straight Shot.csv",
    }
    prefix_map = {"Indoor_Cross": "ICT", "Indoor_Straight": "IST", "Outdoor_Cross": "OCT", "Outdoor_Straight": "OST"}

    all_landing_data = []
    for name, filename in csv_files.items():
        filepath = os.path.join(csv_base_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: Landing CSV not found: {filepath}. Skipping.")
            continue
        try:
            df_landing = pd.read_csv(filepath)
            print(f"  Loaded {len(df_landing)} rows from {filename}")

            df_landing['Environment'] = name.split('_')[0]
            df_landing['ShotType'] = name.split('_')[1]
            df_landing.rename(columns={
                'Index': 'OriginalIndex',
                'To-Closest-Doubles-Sideline-Distance (m)': 'DistSideline',
                'To-Baseline-Distance (m)': 'DistBaseline'
            }, inplace=True, errors='ignore')

            req_cols = ['OriginalIndex', 'DistSideline', 'DistBaseline', 'ShotType']
            if not all(col in df_landing.columns for col in req_cols):
                print(f"Error: Missing required cols in {filename}. Skipping.")
                continue

            file_prefix = prefix_map.get(name, "UNK")
            df_landing['ShotID'] = file_prefix + df_landing['OriginalIndex'].astype(str).str.zfill(2)

            coords = df_landing.apply(lambda row: map_coordinates(row['DistSideline'], row['DistBaseline'], row['ShotType'],
                                                                  DOUBLES_COURT_WIDTH_M, HALF_COURT_LENGTH_M), axis=1)
            df_landing[['NormX', 'NormY']] = pd.DataFrame(coords.tolist(), index=df_landing.index)

            rows_before = len(df_landing)
            df_landing.dropna(subset=['NormX', 'NormY'], inplace=True)
            if len(df_landing) < rows_before:
                 print(f"    Dropped {rows_before - len(df_landing)} rows with invalid coords.")

            final_cols = ['ShotID', 'NormX', 'NormY', 'DistSideline', 'DistBaseline', 'Environment', 'ShotType']
            all_landing_data.append(df_landing[final_cols])
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_landing_data:
        raise RuntimeError("No landing data loaded. Cannot proceed with CNN2.")

    landing_df = pd.concat(all_landing_data, ignore_index=True)
    print(f"\nCombined landing data: {len(landing_df)} valid entries.")
    landing_df.set_index('ShotID', inplace=True, verify_integrity=True)
    print("Landing DataFrame prepared with 'ShotID' index.")
    return landing_df


# --- Frame Sequence Generation (CNN2 Input) ---
def get_sequences_for_cnn2(cnn1_balanced_df, landing_df_indexed, n_frames_sequence_cnn2):
    """Creates frame sequences centered on hit frame and links to landing data."""
    print(f"\nPreparing {n_frames_sequence_cnn2}-frame sequences for CNN2...")
    if cnn1_balanced_df.empty or landing_df_indexed.empty:
        print("Error: Input DataFrames for sequence generation are empty.")
        return []

    sequences_for_dataset = []
    video_groups = cnn1_balanced_df.groupby('video_id')
    skipped_no_hit, skipped_no_landing, skipped_short, skipped_parse = 0, 0, 0, 0

    group_iterator = tqdm(video_groups, desc="Processing Videos for Sequences", total=len(video_groups), ncols=80)
    for video_id, group in group_iterator:
        group = group.copy()
        shot_id = os.path.basename(video_id)

        try: # Find hit frame and sort group by frame number
            group['frame_num_int'] = group['frame_path'].apply(
                lambda x: int(re.search(r'frame_(\d+)', os.path.basename(x)).group(1))
            )
            group_sorted = group.sort_values('frame_num_int')
            sorted_paths = group_sorted['frame_path'].tolist()
            num_total = len(sorted_paths)

            hit_rows = group_sorted[group_sorted['weight'] == 1.0]
            if hit_rows.empty:
                skipped_no_hit += 1
                continue

            hit_row = hit_rows.iloc[len(hit_rows) // 2] # Take middle if multiple
            hit_idx_sorted = sorted_paths.index(hit_row['frame_path'])
        except Exception:
            skipped_parse += 1
            continue

        # Link to landing data
        if shot_id not in landing_df_indexed.index:
            skipped_no_landing += 1
            continue
        target_coords = tuple(landing_df_indexed.loc[shot_id, ['NormX', 'NormY']].values)

        # Extract sequence with padding at ends
        half_window = n_frames_sequence_cnn2 // 2
        start_idx, end_idx = hit_idx_sorted - half_window, hit_idx_sorted + half_window
        seq_paths = [sorted_paths[np.clip(i, 0, num_total - 1)] for i in range(start_idx, end_idx + 1)]

        if len(seq_paths) == n_frames_sequence_cnn2:
            sequences_for_dataset.append({
                'sequence_paths': seq_paths,
                'target_coords': target_coords,
                'video_id': video_id,
                'shot_id': shot_id
            })
        else:
            skipped_short += 1 # Should not happen with np.clip

    print(f"\nSequence Creation Summary:")
    print(f"  Success: {len(sequences_for_dataset)}, Skipped (No Hit): {skipped_no_hit}, "
          f"Skipped (No Landing): {skipped_no_landing}, Skipped (Parse): {skipped_parse}, "
          f"Skipped (Short): {skipped_short}")
    return sequences_for_dataset


def split_sequences(all_sequences, test_size=0.15, val_size=0.15, random_state=42):
    """Splits the list of sequence dictionaries into train/val/test."""
    if not all_sequences: return [], [], []

    # Attempt stratification by shot type prefix
    try:
        strat_labels = [seq['shot_id'][:3] for seq in all_sequences]
        unique_labels, counts = np.unique(strat_labels, return_counts=True)
        can_stratify = np.all(counts >= 2) and len(unique_labels) > 1
        if not can_stratify: print("Warning: Cannot stratify sequence split by shot type.")
    except Exception:
        can_stratify = False
        strat_labels = None
        print("Warning: Failed to create labels for stratification.")

    # Split off Test
    train_val_seq, test_seq = train_test_split(
        all_sequences, test_size=test_size, random_state=random_state,
        stratify=strat_labels if can_stratify else None
    )

    # Split Train/Val
    if len(train_val_seq) < 2:
        print("Warning: Not enough sequences left for train/val split.")
        train_seq, val_seq = train_val_seq, []
    else:
        relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
        strat_labels_tv = [seq['shot_id'][:3] for seq in train_val_seq] if can_stratify else None
        can_stratify_tv = False
        if can_stratify and strat_labels_tv:
             unique_tv, counts_tv = np.unique(strat_labels_tv, return_counts=True)
             can_stratify_tv = np.all(counts_tv >= 2) and len(unique_tv) > 1
             if not can_stratify_tv: print("Warning: Cannot stratify train/val sequence split.")

        train_seq, val_seq = train_test_split(
            train_val_seq, test_size=relative_val_size, random_state=random_state,
            stratify=strat_labels_tv if can_stratify_tv else None
        )

    print(f"\nSequence Splitting Results:")
    print(f"  Training: {len(train_seq)}, Validation: {len(val_seq)}, Test: {len(test_seq)}")
    return train_seq, val_seq, test_seq