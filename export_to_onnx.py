import torch
import torch.nn as nn
import os
import sys

# Import your model components
# Ensure these files are in the current script's search path
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from data.nc import NcDataset # Needed to get data shape


def export_model_to_onnx(model, input_shape, onnx_file_path):
    """
    Exports a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_shape (tuple): The expected input tensor shape for the model
                             (Batch, Sequence, Channel, Height, Width).
                             Example: (1, 6, 1, 64, 64)
        onnx_file_path (str): The path to save the ONNX file.
    """
    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor for ONNX export
    # The input_shape should be (Batch, Sequence, Channel, Height, Width)
    dummy_input = torch.randn(input_shape)

    # Ensure the model and dummy input are on the same device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # For simplicity, we use CPU here
    model.to(device)
    dummy_input = dummy_input.to(device)

    # Export the model
    try:
        torch.onnx.export(model,
                          dummy_input,
                          onnx_file_path,
                          export_params=True,
                          opset_version=11,  # Recommend using a recent opset version
                          do_constant_folding=True,
                          input_names=['input'],   # Name for the input tensor
                          output_names=['output'], # Name for the output tensor
                          dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'}, # Allow batch_size and sequence_length to be dynamic
                                        'output': {0: 'batch_size', 1: 'sequence_length'}})

        print(f"Model successfully exported to: {onnx_file_path}")
    except Exception as e:
        print(f"Error occurred during ONNX export: {e}")

if __name__ == "__main__":
    # --- 1. Determine model parameters and input shape ---
    # This logic needs to be consistent with the model selection logic in main.py
    # Assume convgru is used by default (consistent with the 'else' branch in main.py)
    encoder_params = convgru_encoder_params
    decoder_params = convgru_decoder_params

    # Instantiate the model
    encoder = Encoder(encoder_params[0], encoder_params[1])
    decoder = Decoder(decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)

    # Get an example input to determine the shape
    # To avoid dependency on command-line arguments, we set some default values manually.
    # These values should match the default args.frames_input in your main.py,
    # and the actual output shape of NcDataset needs to match.
    default_frames_input = 6

    # Try to get an actual input shape from NcDataset
    try:
        # An instance of NcDataset is needed here to get the data shape.
        # Note: If NcDataset requires a 'root' parameter, please provide a valid path.
        # We assume it can find the 'data/' directory here.
        temp_dataset = NcDataset(is_train=True, root='data/',
                                 n_frames_input=default_frames_input,
                                 n_frames_output=6) # n_frames_output does not affect input shape here
        # Take the first sample to determine the shape.
        # NcDataset typically returns (idx, targetVar, inputVar)
        _, _, sample_input = temp_dataset[0]
        # The shape of sample_input is usually (Sequence, Channel, Height, Width)
        # We need to add a batch_size dimension to it, so the final shape is (1, S, C, H, W)
        input_shape_for_onnx = (1,) + sample_input.shape
        print(f"Example input shape (S,C,H,W) obtained from NcDataset: {sample_input.shape}")
        print(f"Full input shape (B,S,C,H,W) for ONNX export: {input_shape_for_onnx}")

    except Exception as e:
        print(f"Could not retrieve data shape from NcDataset. Please check the 'data/' path and NcDataset implementation: {e}")
        print("Using a default hardcoded input shape (1, 6, 1, 64, 64). Please adjust according to your actual data!")
        # Fallback to a common shape if NcDataset cannot be loaded
        # You need to modify this based on your actual data channels and image dimensions
        input_shape_for_onnx = (1, default_frames_input, 1, 64, 64) # (B, S, C, H, W)


    # --- 2. Specify ONNX file path ---
    output_onnx_file = 'model_architecture.onnx'

    # --- 3. Perform the export ---
    export_model_to_onnx(net, input_shape_for_onnx, output_onnx_file)

    # --- 4. Suggest using Netron for visualization ---
    print("\n--- Visualize ONNX Model ---")
    print("You can download and use the Netron tool to visualize the exported .onnx file:")
    print("  Download Link: https://netron.app/")
    print(f"  Open the {output_onnx_file} file to view the model architecture graph.")
    print("  Netron displays the type, input/output shapes, and parameter information for each layer, which is very intuitive.")