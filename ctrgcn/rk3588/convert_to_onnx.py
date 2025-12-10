import torch
import torch.onnx
import os
import numpy as np
from ctrgcn_rk3588 import CTRGCN_RK3588
import onnx

def convert_pth_to_onnx(checkpoint_path, onnx_path, input_shape=(1, 3, 30, 17)):
    """
    Convert PyTorch model checkpoint to ONNX format
    
    Args:
        checkpoint_path (str): Path to the .pth checkpoint file
        onnx_path (str): Output path for the .onnx file
        input_shape (tuple): Input tensor shape (batch_size, channels, sequence length, num_joints)
    """
    
    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract config from checkpoint
    config = checkpoint.get('config', {})
    
    # Initialize the model with config parameters
    model = CTRGCN_RK3588(
        num_classes=config.get('num_classes', 2),  # Default to 2 classes (fall/no-fall)
        num_joints=config.get('num_joints', 17),   # Default to 17 joints
        in_channels=3,      # x, y, confidence
        base_channel=16
    )
    
    # Load the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Best accuracy from checkpoint: {checkpoint.get('best_accuracy', 'N/A')}")
    
    # Create dummy input tensor
    dummy_input = torch.randn(*input_shape)
    print(f"Created dummy input with shape: {dummy_input.shape}")
    
    # Export to ONNX
    print(f"Converting to ONNX format...")
    
    torch.onnx.export(
        model,                          # Model to export
        dummy_input,                    # Model input (or a tuple for multiple inputs)
        onnx_path,                      # Output file path
        export_params=True,             # Store the trained parameter weights inside the model file
        opset_version=12,               # ONNX version to export the model to
        do_constant_folding=True,       # Whether to execute constant folding for optimization
        input_names=['input'],          # Model's input names
        output_names=['output'],        # Model's output names
        dynamic_axes={                  # Variable length axes
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model successfully converted to ONNX!")
    print(f"ONNX model saved at: {onnx_path}")
    
    # Verify the conversion
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed!")
        
        # Print model info
        print(f"\nModel Info:")
        print(f"- Inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"- Outputs: {[output.name for output in onnx_model.graph.output]}")
        
    except ImportError:
        print("⚠️  ONNX package not found. Install with: pip install onnx")
    except Exception as e:
        print(f"⚠️  ONNX model verification failed: {e}")

def main():
    # Paths
    checkpoint_path = "checkpoints/best_model.pth"
    onnx_path = "../models/rk3588/ctrgcn.onnx"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Convert the model
    try:
        convert_pth_to_onnx(checkpoint_path, onnx_path)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("Please check your model architecture and input shape.")

if __name__ == "__main__":
    main()