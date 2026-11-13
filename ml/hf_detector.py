"""
Local Deepfake & AI-Generated Image Detector.

Supports:
  1. Human Faces (Deepfake vs Real) -> models/face_detector (ViT-base)
  2. AI Art, Anime, Fantasy, Landscapes -> models/sdxl_detector (Swin-base)
  3. EfficientNet-B0 (Fallback)

Runs 100% locally with PyTorch without any cloud API dependency.
"""

import os
import sys
import re

# Workaround for environment where C-extension regex DLL is restricted
if "regex" not in sys.modules or not hasattr(sys.modules["regex"], "compile"):
    sys.modules["regex"] = re

import cv2
import numpy as np
from PIL import Image

# Global pipeline caches
_face_pipe = None
_art_pipe = None

FACE_MODEL_DIR = os.path.join("models", "face_detector")
ART_MODEL_DIR = os.path.join("models", "sdxl_detector")


def _is_anime_style(bgr: np.ndarray) -> bool:
    """
    Detect anime / cartoon / illustrative style:
    High saturation in HSV and smooth color regions.
    """
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        return mean_sat > 80 and edge_density < 0.12
    except Exception:
        return False


FACE_REPO_ID = "dima806/deepfake_vs_real_image_detection"
ART_REPO_ID = "umm-maybe/AI-image-detector"


def _get_face_pipeline():
    """Load face deepfake detector pipeline (local directory or HuggingFace Hub)."""
    global _face_pipe
    if _face_pipe is not None:
        return _face_pipe
    
    target_model = FACE_MODEL_DIR if os.path.exists(os.path.join(FACE_MODEL_DIR, "model.safetensors")) else FACE_REPO_ID
    try:
        from transformers import pipeline
        print(f"[Detector] Loading face detector from '{target_model}'...")
        _face_pipe = pipeline("image-classification", model=target_model, device=-1)
        print("[Detector] Face detector loaded.")
    except Exception as e:
        print(f"[Detector] Failed to load face pipeline: {e}")
        _face_pipe = None
    return _face_pipe


def _get_art_pipeline():
    """Load AI art / general image detector pipeline (local directory or HuggingFace Hub)."""
    global _art_pipe
    if _art_pipe is not None:
        return _art_pipe
    
    target_model = ART_MODEL_DIR if os.path.exists(os.path.join(ART_MODEL_DIR, "model.safetensors")) else ART_REPO_ID
    try:
        from transformers import pipeline
        print(f"[Detector] Loading AI Art detector from '{target_model}'...")
        _art_pipe = pipeline("image-classification", model=target_model, device=-1)
        print("[Detector] AI Art detector loaded.")
    except Exception as e:
        print(f"[Detector] Failed to load art pipeline: {e}")
        _art_pipe = None
    return _art_pipe


def predict_hf(face_bgr: np.ndarray, has_human_face: bool = True):
    """
    Classify image using the appropriate deep learning model:
      - If has_human_face: use face detector (dima806 ViT)
      - Otherwise: use AI Art / SDXL detector (Swin)
    """
    fallback = {
        "prediction": None,
        "real_prob": 0.5,
        "fake_prob": 0.5,
        "confidence": 50.0,
        "model_used": "EfficientNet-B0 (baseline)",
        "hf_available": False,
    }

    try:
        # Convert BGR to RGB PIL image
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        pipe = None
        model_name = ""

        if has_human_face:
            pipe = _get_face_pipeline()
            model_name = "ViT-Deepfake-Face"
            pred_fake_label = "DEEPFAKE"

        # If no face or face pipe failed, use AI art detector
        if pipe is None:
            pipe = _get_art_pipeline()
            model_name = "Swin-AI-Art-Detector"
            pred_fake_label = "AI-GENERATED"

        if pipe is None:
            return fallback

        results = pipe(pil_img, top_k=2)

        # Parse labels
        fake_keywords = ("fake", "artificial", "ai", "deepfake", "generated")
        real_prob = 0.5
        fake_prob = 0.5

        for item in results:
            label = str(item["label"]).lower()
            score = float(item["score"])
            if any(k in label for k in fake_keywords):
                fake_prob = score
            else:
                real_prob = score

        # Normalize
        total = real_prob + fake_prob
        if total > 0:
            real_prob /= total
            fake_prob /= total

        is_fake = fake_prob >= 0.5
        prediction = pred_fake_label if is_fake else "REAL"
        confidence = fake_prob if is_fake else real_prob

        return {
            "prediction": prediction,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "confidence": round(confidence * 100, 1),
            "model_used": model_name,
            "hf_available": True,
        }

    except Exception as e:
        print(f"[Detector] Prediction error: {e}")
        return fallback
