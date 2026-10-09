"""
Export PyTorch Diabetic Retinopathy Model to ONNX Web/Edge Deployment Format.
"""

import os
import argparse
import torch
from model import get_model

def export_to_onnx(checkpoint_path, output_onnx_path="models/diabetic_retinopathy_resnet50.onnx", backbone="resnet50"):
    device = torch.device("cpu")
    model = get_model(backbone=backbone, num_classes=5, pretrained=False)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from '{checkpoint_path}'")
    else:
        print(f"Notice: Checkpoint '{checkpoint_path}' not found. Exporting random initialized model structure for deployment testing.")
        
    model.eval()
    dummy_input = torch.randn(1, 3, 512, 512, device=device)
    
    os.makedirs(os.path.dirname(output_onnx_path) or ".", exist_ok=True)
    
    torch.onnx.export(
        model.model if hasattr(model, 'model') else model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['class_logits'],
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'class_logits': {0: 'batch_size'}
        }
    )
    print(f"ONNX Model successfully exported to '{output_onnx_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch DR Model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dr_model.pth")
    parser.add_argument("--output", type=str, default="models/dr_model.onnx")
    parser.add_argument("--backbone", type=str, default="resnet50")
    args = parser.parse_args()
    
    export_to_onnx(args.checkpoint, args.output, args.backbone)
