# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import collections # For OrderedDict if needed, but Sequential is fine

# Import defaults for Final model signature
from config import DEFAULT_CNN1_FILTERS, DEFAULT_CNN1_FC_SIZE, DEFAULT_CNN1_DROPOUT, IMG_HEIGHT, IMG_WIDTH

# --- CNN1: Hit Frame Regressor ---

class HitFrameRegressorParam(nn.Module):
    """Parameterized CNN for Hit Frame Regression (Grid Search)."""
    def __init__(self, input_channels=3, img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 block_filters=(32, 64, 128), fc_size=512, dropout_rate=0.5):
        super().__init__()
        # print(f"Initializing HitFrameRegressorParam: filters={block_filters}, fc_size={fc_size}, dropout={dropout_rate}") # Optional debug

        layers = []
        current_channels = input_channels
        current_h, current_w = img_height, img_width

        for i, num_filters in enumerate(block_filters):
            layers.extend([
                nn.Conv2d(current_channels, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            current_channels = num_filters
            current_h //= 2
            current_w //= 2

        self.conv_blocks = nn.Sequential(*layers)
        flattened_size = current_channels * current_h * current_w

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, 1) # Output regression score
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_block(x)
        return x

class HitFrameRegressorFinal(nn.Module):
    """Finalized CNN for Hit Frame Regression (using best found params)."""
    def __init__(self, input_channels=3, img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 block_filters=DEFAULT_CNN1_FILTERS, # Use defaults from config initially
                 fc_size=DEFAULT_CNN1_FC_SIZE,
                 dropout_rate=DEFAULT_CNN1_DROPOUT):
        super().__init__()
        print(f"Initializing HitFrameRegressorFinal:")
        print(f"  - Filters: {block_filters}, FC Size: {fc_size}, Dropout: {dropout_rate}")

        layers = []
        current_channels = input_channels
        current_h, current_w = img_height, img_width

        for i, num_filters in enumerate(block_filters):
            layers.extend([
                nn.Conv2d(current_channels, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            current_channels = num_filters
            current_h //= 2
            current_w //= 2

        self.conv_blocks = nn.Sequential(*layers)
        flattened_size = current_channels * current_h * current_w
        print(f"  - Final Model Flattened Size: {flattened_size}")

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_size, 1)
        )
        print(f"  - Final Model FC Block: {flattened_size} -> {fc_size} -> 1")

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_block(x)
        return x


# --- CNN2: Landing Spot Predictor ---

class LandingPointCNN(nn.Module):
    """CNN for Landing Point Prediction."""
    def __init__(self, input_channels, output_dim=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
        super().__init__()
        print(f"Initializing LandingPointCNN with {input_channels} input channels.")
        # Input shape: [B, input_channels (SeqLen * 3), H, W]

        current_h, current_w = img_height, img_width
        conv_layers = []

        # Block 1
        conv_layers.extend([
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False), # H/2, W/2
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/4, W/4
        ])
        current_h //= 4; current_w //= 4; current_c = 64

        # Block 2
        conv_layers.extend([
            nn.Conv2d(current_c, 128, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/8, W/8
        ])
        current_h //= 2; current_w //= 2; current_c = 128

        # Block 3
        conv_layers.extend([
            nn.Conv2d(current_c, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/16, W/16
        ])
        current_h //= 2; current_w //= 2; current_c = 256

        # Block 4
        conv_layers.extend([
            nn.Conv2d(current_c, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # H/32, W/32
        ])
        current_h //= 2; current_w //= 2; current_c = 512

        self.conv_blocks = nn.Sequential(*conv_layers)
        flattened_size = current_c * current_h * current_w
        print(f"  - CNN2 Flattened size: {flattened_size} ({current_c}x{current_h}x{current_w})")

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, output_dim),
            nn.Sigmoid() # Output normalized coords [0, 1]
        )
        print(f"  - CNN2 FC block: {flattened_size} -> 1024 -> 512 -> {output_dim}")

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_block(x)
        return x