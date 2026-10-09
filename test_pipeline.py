"""
End-to-End Test Script for Diabetic Retinopathy CV & Deep Learning Pipeline.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from preprocess import process_fundus_pipeline
from model import get_model

# ICDR Grade Names & Metadata
ICDR_CLASSES = {
    0: {"name": "Grade 0: No DR", "risk": "Low Risk", "color": "Green"},
    1: {"name": "Grade 1: Mild NPDR", "risk": "Mild Risk", "color": "Yellow"},
    2: {"name": "Grade 2: Moderate NPDR", "risk": "Moderate Risk", "color": "Orange"},
    3: {"name": "Grade 3: Severe NPDR", "risk": "High Risk", "color": "Red"},
    4: {"name": "Grade 4: Proliferative DR", "risk": "Critical Risk", "color": "Crimson"}
}

def run_test_on_image(image_path, model_backbone="resnet50", output_dir="test_output"):
    print(f"\n=======================================================")
    print(f"👁️ Testing Retinopathy AI Model on: {image_path}")
    print(f"=======================================================")

    if not os.path.exists(image_path):
        print(f"Error: Target image file '{image_path}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # STEP 1: Computer Vision Preprocessing Pipeline
    print("\n[Step 1/3] Running Computer Vision Preprocessing Pipeline...")
    cv_outputs = process_fundus_pipeline(image_path, target_size=512)

    # Save CV Preprocessed Variant Artifacts
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_green_channel.png"), cv_outputs["green_channel"])
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_clahe_enhanced.png"), cv2.cvtColor(cv_outputs["clahe_rgb"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_vessel_network.png"), cv_outputs["vessels_enhanced"])
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_ben_graham.png"), cv2.cvtColor(cv_outputs["ben_graham"], cv2.COLOR_RGB2BGR))
    print(f"  [✓] Preprocessed outputs saved to '{output_dir}/'")

    # STEP 2: PyTorch Model Load & Inference
    print("\n[Step 2/3] Loading PyTorch Model & Running Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(backbone=model_backbone, num_classes=5, pretrained=False).to(device)
    model.eval()

    # Prepare PyTorch Image Tensor
    img_rgb = cv_outputs["original_cropped"]
    img_normalized = (img_rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    input_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Forward Pass & Grad-CAM Heatmap
    heatmap_np, logits = model.generate_gradcam(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class]) * 100.0

    meta = ICDR_CLASSES[pred_class]
    print("\n-------------------------------------------------------")
    print(f"🎯 AI Screening Result: {meta['name']}")
    print(f"⚠️  Risk Category:     {meta['risk']}")
    print(f"📊 Confidence Score:  {confidence:.2f}%")
    print("-------------------------------------------------------")
    print("Multi-Class Probability Breakdown:")
    for idx, p in enumerate(probs):
        bar = "█" * int(p * 25)
        print(f"  Class {idx} [{ICDR_CLASSES[idx]['name']:<24}]: {p*100:6.2f}% | {bar}")

    # STEP 3: Grad-CAM Explainability Overlay Generation
    print("\n[Step 3/3] Generating Explainable AI (Grad-CAM) Heatmap...")
    heatmap_resized = cv2.resize(heatmap_np, (512, 512))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay Heatmap on Cropped Original Fundus
    orig_bgr = cv2.cvtColor(cv_outputs["original_cropped"], cv2.COLOR_RGB2BGR)
    gradcam_overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    overlay_save_path = os.path.join(output_dir, f"{base_name}_gradcam_overlay.png")
    cv2.imwrite(overlay_save_path, gradcam_overlay)
    print(f"  [✓] Grad-CAM overlay heatmap saved to: '{overlay_save_path}'")
    print("=======================================================\n")

if __name__ == "__main__":
    test_image = "test_samples/grade2_moderate.png"
    if not os.path.exists(test_image):
        print(f"Generating test images first...")
        os.system("python python_pipeline/generate_test_images.py")
        
    run_test_on_image(test_image)
