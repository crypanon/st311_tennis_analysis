# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import collections # For OrderedDict if needed, but Sequential is fine

# Import defaults for Final model signature
from config import (DEFAULT_CNN1_FILTERS, DEFAULT_CNN1_FC_SIZE, DEFAULT_CNN1_DROPOUT,
                    DEFAULT_CNN2_CONV_FILTERS, DEFAULT_CNN2_FC_SIZES, DEFAULT_CNN2_DROPOUT, # Added CNN2 defaults
                    IMG_HEIGHT, IMG_WIDTH)
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


class LandingPointCNNParam(nn.Module):
    """Parameterized CNN for Landing Point Prediction (Grid Search)."""
    def __init__(self, input_channels, output_dim=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 conv_filters=(64, 128, 256, 512), # List/tuple of filters per block
                 fc_sizes=(1024, 512),             # List/tuple of FC layer sizes
                 dropout_rate=0.5):
        super().__init__()
        # print(f"Init LandingPointCNNParam: conv={conv_filters}, fc={fc_sizes}, drop={dropout_rate}") # Debug

        current_h, current_w = img_height, img_width
        conv_layers = []
        current_c = input_channels

        # Define kernel sizes/strides/padding (keep these fixed for simplicity, like original)
        # Block 1: 7x7 conv, stride 2, pool 3x3 stride 2
        # Block 2: 5x5 conv, stride 1, pool 3x3 stride 2
        # Block 3: 3x3 conv, stride 1, pool 3x3 stride 2
        # Block 4: 3x3 conv, stride 1, pool 3x3 stride 2
        # You could parameterize these too, but it gets complex quickly.
        block_configs = [
            {'k': 7, 's': 2, 'p': 3, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1}, # Block 1
            {'k': 5, 's': 1, 'p': 2, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1}, # Block 2
            {'k': 3, 's': 1, 'p': 1, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1}, # Block 3
            {'k': 3, 's': 1, 'p': 1, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1}  # Block 4
        ]

        # Dynamically build conv blocks based on conv_filters length
        num_blocks = len(conv_filters)
        for i in range(num_blocks):
            out_c = conv_filters[i]
            # Use corresponding config or last one if filters list is longer/shorter
            cfg = block_configs[min(i, len(block_configs)-1)]

            conv_layers.extend([
                nn.Conv2d(current_c, out_c, kernel_size=cfg['k'], stride=cfg['s'], padding=cfg['p'], bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            ])
            # Calculate spatial changes from conv stride
            current_h = (current_h + 2*cfg['p'] - cfg['k']) // cfg['s'] + 1
            current_w = (current_w + 2*cfg['p'] - cfg['k']) // cfg['s'] + 1

            # Add pooling layer
            conv_layers.append(
                nn.MaxPool2d(kernel_size=cfg['pool_k'], stride=cfg['pool_s'], padding=cfg['pool_p'])
            )
            # Calculate spatial changes from pool stride
            current_h = (current_h + 2*cfg['pool_p'] - cfg['pool_k']) // cfg['pool_s'] + 1
            current_w = (current_w + 2*cfg['pool_p'] - cfg['pool_k']) // cfg['pool_s'] + 1

            current_c = out_c # Update channel count

        self.conv_blocks = nn.Sequential(*conv_layers)
        flattened_size = current_c * current_h * current_w
        # print(f"  - Param CNN2 Flattened: {flattened_size} ({current_c}x{current_h}x{current_w})") # Debug

        # Dynamically build FC layers
        fc_layers = [nn.Flatten()]
        last_size = flattened_size
        for size in fc_sizes:
            fc_layers.extend([
                nn.Linear(last_size, size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            last_size = size
        # Final output layer
        fc_layers.extend([
            nn.Linear(last_size, output_dim),
            nn.Sigmoid() # Output normalized coords [0, 1]
        ])
        self.fc_block = nn.Sequential(*fc_layers)
        # print(f"  - Param CNN2 FC block built.") # Debug

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_block(x)
        return x


# --- Update standard LandingPointCNN to accept arch params ---
class LandingPointCNN(nn.Module): # Rename/replace original
    """Finalized CNN for Landing Point Prediction."""
    def __init__(self, input_channels, output_dim=2, img_height=IMG_HEIGHT, img_width=IMG_WIDTH,
                 conv_filters=DEFAULT_CNN2_CONV_FILTERS, # Use defaults from config
                 fc_sizes=DEFAULT_CNN2_FC_SIZES,
                 dropout_rate=DEFAULT_CNN2_DROPOUT):
        super().__init__()
        print(f"Initializing LandingPointCNN:")
        print(f"  - Input Channels: {input_channels}")
        print(f"  - Conv Filters: {conv_filters}")
        print(f"  - FC Sizes: {fc_sizes}")
        print(f"  - Dropout: {dropout_rate}")

        current_h, current_w = img_height, img_width
        conv_layers = []
        current_c = input_channels

        # Fixed block configs (same as Param model)
        block_configs = [
            {'k': 7, 's': 2, 'p': 3, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1},
            {'k': 5, 's': 1, 'p': 2, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1},
            {'k': 3, 's': 1, 'p': 1, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1},
            {'k': 3, 's': 1, 'p': 1, 'pool_k': 3, 'pool_s': 2, 'pool_p': 1}
        ]
        num_blocks = len(conv_filters)
        for i in range(num_blocks):
            out_c = conv_filters[i]
            cfg = block_configs[min(i, len(block_configs)-1)]
            conv_layers.extend([
                nn.Conv2d(current_c, out_c, kernel_size=cfg['k'], stride=cfg['s'], padding=cfg['p'], bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            ])
            current_h = (current_h + 2*cfg['p'] - cfg['k']) // cfg['s'] + 1
            current_w = (current_w + 2*cfg['p'] - cfg['k']) // cfg['s'] + 1
            conv_layers.append(
                nn.MaxPool2d(kernel_size=cfg['pool_k'], stride=cfg['pool_s'], padding=cfg['pool_p'])
            )
            current_h = (current_h + 2*cfg['pool_p'] - cfg['pool_k']) // cfg['pool_s'] + 1
            current_w = (current_w + 2*cfg['pool_p'] - cfg['pool_k']) // cfg['pool_s'] + 1
            current_c = out_c

        self.conv_blocks = nn.Sequential(*conv_layers)
        flattened_size = current_c * current_h * current_w
        print(f"  - Final CNN2 Flattened size: {flattened_size} ({current_c}x{current_h}x{current_w})")

        # Dynamically build FC layers
        fc_layers = [nn.Flatten()]
        last_size = flattened_size
        for size in fc_sizes:
            fc_layers.extend([
                nn.Linear(last_size, size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            last_size = size
        fc_layers.extend([
            nn.Linear(last_size, output_dim),
            nn.Sigmoid()
        ])
        self.fc_block = nn.Sequential(*fc_layers)
        print(f"  - Final CNN2 FC block built.")

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_block(x)
        return x