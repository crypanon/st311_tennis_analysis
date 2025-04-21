import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_json_params(config_path, param_name="parameters"):
    """Loads parameters from a JSON file."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                params = json.load(f)
                # Handle potential list->tuple conversion for filters
                if 'filters' in params and isinstance(params['filters'], list):
                    params['filters'] = tuple(params['filters'])
                print(f"Loaded {param_name} from: {os.path.basename(config_path)}")
                return params
        except Exception as e:
            print(f"Warning: Error loading {param_name} from {os.path.basename(config_path)}: {e}. Returning None.")
            return None
    else:
        print(f"Warning: {param_name} file not found: {config_path}. Returning None.")
        return None

def plot_training_history(history_data, model_name, save_path=None):
    """Plots training and validation loss and validation MAE."""
    if isinstance(history_data, str) and os.path.exists(history_data):
        try:
            history_df = pd.read_csv(history_data)
            history = history_df.to_dict(orient='list')
            print(f"Loaded history for {model_name} from CSV.")
        except Exception as e:
            print(f"Error loading history CSV {history_data}: {e}")
            return
    elif isinstance(history_data, dict):
        history = history_data
    else:
        print(f"Invalid history data for {model_name}. Cannot plot.")
        return

    if not history or 'train_loss' not in history or 'val_loss' not in history:
        print(f"History data for {model_name} is missing required keys ('train_loss', 'val_loss').")
        return

    epochs_trained = len(history['train_loss'])
    if epochs_trained == 0:
        print(f"No epochs recorded in history for {model_name}.")
        return

    epoch_range = range(1, epochs_trained + 1)
    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f"Final Training History: {model_name}", fontsize=16)

    # Plot Loss
    axes[0].plot(epoch_range, history['train_loss'], marker='.', linestyle='-', label='Training Loss (MSE)')
    axes[0].plot(epoch_range, history['val_loss'], marker='.', linestyle='-', label='Validation Loss (MSE)')
    axes[0].set_title('Model Loss (MSE)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Ensure integer ticks

    # Plot MAE
    if 'val_mae' in history and history['val_mae'] and not np.all(pd.isna(history['val_mae'])):
        axes[1].plot(epoch_range, history['val_mae'], marker='.', linestyle='-', label='Validation MAE', color='green')
        axes[1].set_title('Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].legend()
        axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
         axes[1].set_title('Validation MAE (No Data)')
         axes[1].text(0.5, 0.5, 'MAE data not available', horizontalalignment='center', verticalalignment='center', transform=axes[1].transAxes)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Saved training plot to: {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.show()


def setup_kaggle_downloader():
    """Attempts to login and download the dataset using Kaggle Hub."""
    try:
        import kagglehub
        kagglehub.login() # Uses stored credentials
        print("Kaggle login successful (using stored credentials).")
        print("Attempting to download/verify dataset 'joekinder/st311-tennis'...")
        # Set download path relative to project root if possible, or let kagglehub decide
        # download_dir = os.path.join(config.PROJECT_ROOT, 'kaggle_datasets')
        # os.makedirs(download_dir, exist_ok=True)
        # dataset_path = kagglehub.dataset_download("joekinder/st311-tennis", path=download_dir)
        dataset_path = kagglehub.dataset_download("joekinder/st311-tennis") # Use default cache loc
        print(f"Dataset downloaded/verified at: {dataset_path}")
        if not os.path.exists(dataset_path):
             raise FileNotFoundError(f"Kaggle dataset path not found after download: {dataset_path}")
        return dataset_path
    except ImportError:
         print("Warning: 'kagglehub' not installed. Cannot automatically download.")
         print("Please install it (`pip install kagglehub`) or download manually.")
         return None
    except Exception as e:
        print(f"Error during Kaggle login/download: {e}")
        print("Please ensure Kaggle API key is configured and dataset exists.")
        return None