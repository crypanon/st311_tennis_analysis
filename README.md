# ST311 Tennis Analysis Project

## Overview
This project focuses on analysing tennis data using Convolutional Neural Networks (CNNs). It includes two primary tasks:
1. **Hit Frame Regression**: Predicting the frame where a tennis ball is hit.
2. **Landing Spot Prediction**: Estimating the landing coordinates of the ball.

The project is implemented in Python using PyTorch and includes various utilities for data preprocessing, model training, and evaluation.

## Features
- **Data Preprocessing**: Utilities to load and preprocess tennis match data, including frame sequences and landing coordinates.
- **Model Training**: Scripts for training CNN models with support for hyperparameter tuning and early stopping.
- **Prediction Pipeline**: Functions to predict hit frames and landing spots using trained models.
- **Visualisation**: Tools to visualise predictions and training progress.

## Project Structure
- `main.py`: Entry point for running different components of the project.
- `config.py`: Configuration file for paths, hyperparameters, and other settings.
- `data_utils.py`: Functions for loading and preprocessing data.
- `training.py`: Scripts for training and evaluating models.
- `models.py`: Definitions of CNN architectures used in the project.
- `st311_project_output/`: Directory containing model weights, training history, and results.

## Getting Started

### Prerequisites
- Python 3.8 or later
- PyTorch
- Additional dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd st311_tennis_analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the main script with appropriate arguments:
   ```bash
   python main.py --run_grid_search
   ```
2. Train models or evaluate predictions using the provided scripts.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.
