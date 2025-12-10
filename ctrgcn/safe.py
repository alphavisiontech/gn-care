import torch
import numpy as np

def convert_checkpoint_to_safe(old_checkpoint_path, new_checkpoint_path):
    """Convert existing checkpoint to NumPy-safe format"""
    print(f"🔧 Converting checkpoint: {old_checkpoint_path}")
    
    # Load with unsafe method
    torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar', 'numpy.ndarray'])
    old_checkpoint = torch.load(old_checkpoint_path, weights_only=False)
    
    def numpy_to_python(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: numpy_to_python(value) for key, value in obj.items()}
        else:
            return obj
    
    # Convert all NumPy objects
    new_checkpoint = numpy_to_python(old_checkpoint)
    
    # Save safe version
    torch.save(new_checkpoint, new_checkpoint_path)
    print(f"✅ Safe checkpoint saved: {new_checkpoint_path}")

if __name__ == "__main__":
    convert_checkpoint_to_safe('./models/v11/best_model.pth', './models/v11/best_model_safe.pth')