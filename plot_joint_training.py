import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math # For checking nan/inf

# --- Configuration ---
# Path to the project output directory (where the CSV is saved)
# Adjust this if your script is not in the same root as the project output folder
PROJECT_OUTPUT_PATH = './st311_project_output/' # Or './' if script is inside st311_project_output

HISTORY_CSV_FILENAME = 'joint_final_training_history.csv'
PLOT_SAVE_FILENAME = 'joint_final_training_plot.png'

history_csv_path = os.path.join(PROJECT_OUTPUT_PATH, HISTORY_CSV_FILENAME)
plot_save_path = os.path.join(PROJECT_OUTPUT_PATH, PLOT_SAVE_FILENAME)

# --- Function to Plot ---
def plot_joint_training_history(csv_path, save_path=None):
    """
    Loads joint training history from a CSV and generates plots.

    Args:
        csv_path (str): Path to the joint_final_training_history.csv file.
        save_path (str, optional): Path to save the generated plot. Defaults to None.
    """
    if not os.path.exists(csv_path):
        print(f"Error: History file not found at {csv_path}")
        return

    try:
        history_df = pd.read_csv(csv_path)
        print(f"Loaded history from: {csv_path}")
        # print("Columns found:", history_df.columns.tolist()) # Debug columns
        # print(history_df.head()) # Debug data
    except Exception as e:
        print(f"Error loading or parsing CSV file: {e}")
        return

    # Check for essential columns
    required_train_cols = ['epoch', 'train_loss', 'train_loss_cnn1', 'train_loss_cnn2', 'train_loss_penalty']
    required_val_cols = ['val_loss_cnn2', 'val_mae_cnn2'] # If validation was run

    if not all(col in history_df.columns for col in required_train_cols):
        print(f"Error: CSV is missing one or more required training columns: {required_train_cols}")
        return

    validation_ran = all(col in history_df.columns for col in required_val_cols) and \
                     not history_df[required_val_cols].isnull().all().all() # Check if val cols exist and are not all NaN

    epochs_trained = len(history_df)
    if epochs_trained == 0:
        print("No epochs found in history data.")
        return

    # Use the 'epoch' column if it exists and starts from 0, otherwise create range
    if 'epoch' in history_df.columns and history_df['epoch'].iloc[0] == 0:
         epoch_range = history_df['epoch'] + 1 # Shift to start from 1 for plotting
    else:
         epoch_range = range(1, epochs_trained + 1)


    plt.style.use('seaborn-v0_8-darkgrid')

    # --- Plot 1: Overall Training Loss ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_range, history_df['train_loss'], marker='.', label='Total Train Loss')
    if validation_ran:
        plt.plot(epoch_range, history_df['val_loss_cnn2'], marker='.', linestyle='--', label='Validation Loss (CNN2 Only)')
    plt.title('Overall Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, min_n_ticks=5))
    if save_path:
        overall_loss_path = save_path.replace('.png', '_overall_loss.png')
        plt.savefig(overall_loss_path, dpi=300)
        print(f"Overall Loss plot saved to: {overall_loss_path}")
    plt.show()
    plt.close()

    # --- Plot 2: Training Loss Components ---
    plt.figure(figsize=(8, 6))
    plt.plot(epoch_range, history_df['train_loss_cnn1'], marker='.', label='Train Loss CNN1 (vs h(x))', alpha=0.8)
    plt.plot(epoch_range, history_df['train_loss_cnn2'], marker='.', label='Train Loss CNN2 (Landing)', alpha=0.8)
    ax1_twin = plt.gca().twinx()
    ax1_twin.plot(epoch_range, history_df['train_loss_penalty'], marker='.', linestyle=':', color='red', label='Train Penalty Loss (MAE Idx Diff)', alpha=0.7)
    plt.title('Training Loss Components vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (CNN1, CNN2)')
    ax1_twin.set_ylabel('MAE Index Difference (Penalty)', color='red')
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, min_n_ticks=5))
    if save_path:
        components_loss_path = save_path.replace('.png', '_components_loss.png')
        plt.savefig(components_loss_path, dpi=300)
        print(f"Training Loss Components plot saved to: {components_loss_path}")
    plt.show()
    plt.close()

    # --- Plot 3: Validation Metrics (CNN2) ---
    if validation_ran:
        plt.figure(figsize=(8, 6))
        plt.plot(epoch_range, history_df['val_loss_cnn2'], marker='.', label='Validation Loss (MSE)')
        ax2_twin = plt.gca().twinx()
        ax2_twin.plot(epoch_range, history_df['val_mae_cnn2'], marker='.', linestyle=':', color='green', label='Validation MAE (Norm. Coords)')
        plt.title('Validation Metrics (CNN2) vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        ax2_twin.set_ylabel('MAE (Normalized Coords)', color='green')
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True, min_n_ticks=5))
        if save_path:
            validation_metrics_path = save_path.replace('.png', '_validation_metrics.png')
            plt.savefig(validation_metrics_path, dpi=300)
            print(f"Validation Metrics plot saved to: {validation_metrics_path}")
        plt.show()
        plt.close()
    else:
        print("Validation metrics not available, skipping validation plot.")


# --- Main Execution ---
if __name__ == "__main__":
    plot_joint_training_history(history_csv_path, plot_save_path)