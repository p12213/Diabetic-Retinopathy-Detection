"""
Diabetic Retinopathy PyTorch Model Architecture Module
Supports ResNet-50 and EfficientNet backbones with custom classification head and Grad-CAM hooks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DiabeticRetinopathyClassifier(nn.Module):
    def __init__(self, backbone="resnet50", num_classes=5, pretrained=True, dropout_rate=0.4):
        """
        Backbone choices: 'resnet50', 'efficientnet_b4', 'mobilenet_v3'
        num_classes: 5 (ICDR standard: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR)
        """
        super(DiabeticRetinopathyClassifier, self).__init__()
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            # Replace final classification head
            self.model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
            self.target_layer = self.model.layer4[-1]
            
        elif backbone == "efficientnet_b4":
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            self.model = models.efficientnet_b4(weights=weights)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
            self.target_layer = self.model.features[-1]
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'efficientnet_b4'.")
            
        # Placeholders for Grad-CAM gradients and activations
        self.gradients = None
        self.activations = None
        self._register_gradcam_hooks()

    def _register_gradcam_hooks(self):
        """Registers backward and forward hooks on the target convolutional layer for Grad-CAM."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x):
        return self.model(x)

    def generate_gradcam(self, x, class_idx=None):
        """
        Generates Grad-CAM heatmap tensor for input image batch x.
        """
        self.eval()
        output = self.forward(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input dimensions
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        return cam.squeeze().cpu().numpy(), output

def get_model(backbone="resnet50", num_classes=5, pretrained=True):
    return DiabeticRetinopathyClassifier(backbone=backbone, num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 512, 512)
    model = get_model("resnet50", num_classes=5, pretrained=False)
    out = model(dummy_input)
    print(f"Model output shape: {out.shape} (Expected: [1, 5])")
