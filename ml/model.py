import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B0 based Transfer Learning Model for Deepfake Image Detection.
    Binary Classifier:
    - Index 0: REAL
    - Index 1: DEEPFAKE
    """
    def __init__(self, num_classes=2, pretrained=True, dropout_rate=0.4):
        super(DeepfakeDetector, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Extract input features dimension from the original classifier head
        in_features = self.backbone.classifier[1].in_features
        
        # Replace the final classification head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

    def get_target_layer(self):
        """
        Returns the last convolutional layer of EfficientNet-B0 for Grad-CAM.
        For EfficientNet-B0, features[-1] (Conv2dNormActivation) is the last conv block.
        """
        return self.backbone.features[-1]

def build_model(num_classes=2, pretrained=True, device='cpu'):
    """
    Factory function to instantiate and move model to target device.
    """
    model = DeepfakeDetector(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model

def save_checkpoint(model, filepath, config=None):
    """
    Save model weights along with configuration metadata.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': config or {
            'architecture': 'efficientnet_b0',
            'num_classes': 2,
            'class_names': ['REAL', 'DEEPFAKE'],
            'image_size': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    torch.save(checkpoint, filepath)
    
    # Save config JSON separately for quick inspection
    config_path = os.path.join(os.path.dirname(filepath), 'config.json')
    with open(config_path, 'w') as f:
        json.dump(checkpoint['config'], f, indent=2)
    print(f"[Model] Saved checkpoint to {filepath} and config to {config_path}")

def load_checkpoint(filepath, device='cpu'):
    """
    Load model weights and metadata from checkpoint file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint.get('config', {})
    
    model = build_model(
        num_classes=config.get('num_classes', 2),
        pretrained=False,
        device=device
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, config
