import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_flops_qrqac_action_value(board_width, board_height, quantiles):
    # Model dimensions
    input_channels = 128  # Output channels from the common layers

    # Conv2D (val_conv1)
    val_conv1_out_channels = 2
    val_conv1_kernel_size = 1
    val_conv1_flops = (board_width * board_height *
                       input_channels * val_conv1_out_channels * val_conv1_kernel_size**2)

    # First Linear Layer (val_fc1)
    val_fc1_in_features = 2 * board_width * board_height
    val_fc1_out_features = 64
    val_fc1_flops = val_fc1_in_features * val_fc1_out_features

    # Second Linear Layer (val_fc2)
    val_fc2_in_features = 64
    val_fc2_out_features = board_width * board_height * quantiles
    val_fc2_flops = val_fc2_in_features * val_fc2_out_features

    # Total FLOPs for action value layers
    total_flops = val_conv1_flops + val_fc1_flops + val_fc2_flops

    return total_flops

# Test the function with different quantile values
board_width = 4
board_height = 9
quantiles_list = [3, 9, 27, 81]

for quantiles in quantiles_list:
    flops = compute_flops_qrqac_action_value(board_width, board_height, quantiles)
    print(f"Quantiles: {quantiles}, FLOPs: {flops}")
