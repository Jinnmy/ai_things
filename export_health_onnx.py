import torch
import torch.nn as nn
from health_predictor import HealthPredictor, Autoencoder
import sys
import os

# Set encoding for Windows console to avoid UnicodeEncodeError
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def export_health_model_onnx(model_path='health_model_v2.pth', onnx_path='health_model.onnx'):
    # Initialize predictor to get dimensions and model
    predictor = HealthPredictor(model_path=model_path)
    model = predictor.model
    model.eval()

    # Create dummy input (1, 14) for tracing
    dummy_input = torch.randn(1, 14)

    # Export
    # We don't want external data. For a model this small, it should be internal by default.
    # But we can try to force it if needed by using the newer ONNX API or just standard export.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Check if a .data file was created and merge it if it was (unlikely for this size)
    data_path = onnx_path + ".data"
    if os.path.exists(data_path):
        print(f"Warning: {data_path} was created. This model should be small enough to be internal.")
        # If it's small, we should try to reload and save again without external data
        import onnx
        m = onnx.load(onnx_path)
        onnx.save(m, onnx_path, save_as_external_data=False)
        if os.path.exists(data_path):
            os.remove(data_path)
            print(f"Merged external data and removed {data_path}")

    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    export_health_model_onnx()
