import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    for PyTorch Convolutional Networks (EfficientNet-B0).
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.get_target_layer()
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        Generate normalized Grad-CAM heatmap for a target class index (1 = DEEPFAKE).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        gradients = self.gradients[0].cpu().data.numpy() # [C, H, W]
        activations = self.activations[0].cpu().data.numpy() # [C, H, W]
        
        # Global average pooling of gradients per channel
        weights = np.mean(gradients, axis=(1, 2)) # [C]
        
        # Weighted combination of forward activation maps
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        # Apply ReLU to retain positive influences
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize between 0 and 1
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
        return heatmap, target_class, output

def create_heatmap_overlay(face_bgr, heatmap, alpha=0.5):
    """
    Resize heatmap to face image dimensions and blend with BGR face image.
    """
    h, w, _ = face_bgr.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap float (0..1) to uint8 (0..255)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original face image
    overlay = cv2.addWeighted(face_bgr, 1 - alpha, colormap, alpha, 0)
    
    return heatmap_resized, colormap, overlay

def analyze_activation_regions(heatmap_resized):
    """
    Analyze spatial distribution of Grad-CAM activations across anatomical face regions.
    Returns:
    - top_regions: list of dicts with region_name, score, severity, description
    - evidence_reasons: list of formatted string reasons
    - dominant_region: string name of primary suspicious region
    """
    h, w = heatmap_resized.shape
    
    # Define bounding boxes for facial sub-regions (relative percentages)
    regions_def = {
        'Eyes & Eyebrows': {
            'y_min': int(h * 0.22), 'y_max': int(h * 0.48),
            'x_min': int(w * 0.15), 'x_max': int(w * 0.85),
            'desc': 'Abnormal texture and lighting artifacts around the eye region',
            'severity': 'Critical'
        },
        'Mouth & Lips': {
            'y_min': int(h * 0.65), 'y_max': int(h * 0.88),
            'x_min': int(w * 0.25), 'x_max': int(w * 0.75),
            'desc': 'Unnatural lip contour and mouth cavity boundary artifacts',
            'severity': 'High'
        },
        'Cheeks & Skin Texture': {
            'y_min': int(h * 0.42), 'y_max': int(h * 0.68),
            'x_min': int(w * 0.15), 'x_max': int(w * 0.85),
            'desc': 'Unnatural skin texture smoothing and localized blend anomalies',
            'severity': 'High'
        },
        'Facial Boundary': {
            'y_min': 0, 'y_max': h,
            'x_min': 0, 'x_max': w,
            'mask_type': 'boundary',
            'desc': 'High activation around facial boundaries and chin/jaw contour',
            'severity': 'Critical'
        },
        'Hair & Forehead Line': {
            'y_min': 0, 'y_max': int(h * 0.25),
            'x_min': int(w * 0.10), 'x_max': int(w * 0.90),
            'desc': 'Hairline and upper forehead boundary integration inconsistencies',
            'severity': 'Medium'
        }
    }
    
    region_scores = {}
    
    for rname, rdef in regions_def.items():
        if rdef.get('mask_type') == 'boundary':
            # Outer 12% border mask
            b_y = int(h * 0.12)
            b_x = int(w * 0.12)
            mask = np.ones((h, w), dtype=bool)
            mask[b_y:h-b_y, b_x:w-b_x] = False
            region_val = np.mean(heatmap_resized[mask]) if np.any(mask) else 0.0
        else:
            y1, y2 = rdef['y_min'], rdef['y_max']
            x1, x2 = rdef['x_min'], rdef['x_max']
            patch = heatmap_resized[y1:y2, x1:x2]
            region_val = np.mean(patch) if patch.size > 0 else 0.0
            
        region_scores[rname] = {
            'score': float(region_val),
            'desc': rdef['desc'],
            'severity': rdef['severity']
        }
        
    # Sort regions by activation score descending
    sorted_regions = sorted(region_scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    top_activated = []
    evidence_reasons = []
    
    threshold = 0.25 # Minimum activation threshold to flag region
    for rname, rinfo in sorted_regions:
        if rinfo['score'] >= threshold:
            top_activated.append({
                'region': rname,
                'score': round(rinfo['score'], 3),
                'severity': rinfo['severity'],
                'description': rinfo['desc']
            })
            evidence_reasons.append(f"{rinfo['desc']} (Model Evidence Confidence: {int(rinfo['score']*100)}%)")
            
    # Fallback requirement if no specific region strongly dominates
    if not evidence_reasons:
        evidence_reasons.append("Model detected suspicious facial features, but could not determine a specific manipulation artifact.")
        dominant_region = "Diffuse Suspicious Features"
    else:
        dominant_region = sorted_regions[0][0]
        
    return top_activated, evidence_reasons, dominant_region
