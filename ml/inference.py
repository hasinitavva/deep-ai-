import os
import cv2
import time
import uuid
import numpy as np
import torch

from ml.model import build_model, load_checkpoint
from ml.gradcam import GradCAM, create_heatmap_overlay, analyze_activation_regions
from ml.frequency_analysis import analyze_frequency_domain
from ml.hf_detector import predict_hf, _is_anime_style


class FaceDetector:
    """
    OpenCV Haar Cascade face detector.
    Returns detected human face crops; sets 'has_human_face'=False if none found
    so the full image is used instead (for AI art, anime, etc.).
    """
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path) if os.path.exists(cascade_path) else None

    def detect_faces(self, image_bgr):
        """
        Returns list of face dicts.  Each dict has:
            box, confidence, crop_bgr, has_human_face
        If no human face found, returns ONE entry for the full image
        with has_human_face=False.
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        detected = []

        if self.face_cascade is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            for (x, y, fw, fh) in faces:
                pad_x, pad_y = int(fw * 0.15), int(fh * 0.15)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
                detected.append({
                    'box': [x1, y1, x2 - x1, y2 - y1],
                    'confidence': 0.95,
                    'crop_bgr': image_bgr[y1:y2, x1:x2].copy(),
                    'has_human_face': True,
                })

        # No human face — analyse full image (AI art, anime, landscapes, etc.)
        if not detected:
            detected.append({
                'box': [0, 0, w, h],
                'confidence': 0.0,
                'crop_bgr': image_bgr.copy(),
                'has_human_face': False,
            })

        return detected


class DeepfakeInferenceEngine:
    """
    Universal AI-image detector supporting:
      • Deepfake human faces
      • AI-generated art / fantasy images
      • Anime / cartoon AI art
    Uses HuggingFace Inference API as the primary classifier and
    EfficientNet-B0 Grad-CAM for visual heatmap generation.
    """
    def __init__(self, checkpoint_path='models/deepfake_detector.pth', output_dir='outputs/heatmaps'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.face_detector = FaceDetector()
        self.is_checkpoint_loaded = False
        self._init_model()

    def _init_model(self):
        if os.path.exists(self.checkpoint_path):
            try:
                self.model, self.config = load_checkpoint(self.checkpoint_path, device=self.device)
                self.is_checkpoint_loaded = True
                print(f"[InferenceEngine] Loaded checkpoint: '{self.checkpoint_path}'")
            except Exception as e:
                print(f"[InferenceEngine] Checkpoint load warning: {e}. Using baseline.")
                self._create_baseline_model()
        else:
            print(f"[InferenceEngine] No checkpoint found. Using baseline.")
            self._create_baseline_model()

        self.gradcam = GradCAM(self.model)

    def _create_baseline_model(self):
        self.model = build_model(num_classes=2, pretrained=True, device=self.device)
        self.model.eval()
        self.config = {
            'architecture': 'efficientnet_b0',
            'num_classes': 2,
            'class_names': ['REAL', 'AI-GENERATED'],
            'image_size': 224,
            'is_baseline': True
        }
        self.is_checkpoint_loaded = False

    def preprocess(self, bgr, img_size=224):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (img_size, img_size))
        normalized = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        return torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _image_type_label(self, crop_bgr, has_human_face):
        """Return a human-readable label for what type of content was detected."""
        if has_human_face:
            return "Human Face"
        if _is_anime_style(crop_bgr):
            return "Anime / Cartoon"
        return "AI Art / Photo"

    def _build_evidence(self, hf_result, heatmap_regions, freq_score, freq_evidence,
                        prediction, confidence_pct, image_type):
        """Build the explanation reasons list."""
        reasons = []
        is_fake = prediction not in ("REAL",)

        if hf_result["hf_available"]:
            model_short = hf_result["model_used"].split("/")[-1]
            if is_fake:
                reasons.append(
                    f"AI detection model ({model_short}) classified this {image_type} "
                    f"as AI-GENERATED / FAKE with {confidence_pct}% confidence."
                )
            else:
                reasons.append(
                    f"AI detection model ({model_short}) classified this {image_type} "
                    f"as REAL with {confidence_pct}% confidence."
                )

        # Add Grad-CAM spatial regions (relevant for faces)
        if heatmap_regions:
            reasons.extend(heatmap_regions[:2])

        # FFT anomaly evidence
        if is_fake and freq_score > 0.35:
            reasons.append(freq_evidence)
        elif not is_fake:
            reasons.append("No high-frequency grid artifacts were detected in the frequency spectrum.")

        # Fallback if HF not available
        if not hf_result["hf_available"]:
            reasons.append(
                "⚠️ Online AI detection API was unreachable. "
                "Result is from EfficientNet-B0 baseline (may be less accurate for non-face images)."
            )

        return reasons

    def analyze_image(self, image_bgr):
        """
        Full analysis pipeline. Works for:
          - Human face photos (deepfake detection)
          - AI-generated art (fantasy, digital art)
          - Anime / cartoon images
        """
        face_infos = self.face_detector.detect_faces(image_bgr)
        results = []

        for idx, face_info in enumerate(face_infos):
            crop_bgr       = face_info['crop_bgr']
            box            = face_info['box']
            face_conf      = face_info['confidence']
            has_human_face = face_info['has_human_face']

            image_type = self._image_type_label(crop_bgr, has_human_face)

            # ── 1. Primary prediction via HuggingFace API ──────────────────
            hf_result = predict_hf(crop_bgr, has_human_face=has_human_face)

            if hf_result["hf_available"]:
                prediction     = hf_result["prediction"]
                real_prob      = hf_result["real_prob"]
                fake_prob      = hf_result["fake_prob"]
                confidence_pct = hf_result["confidence"]
            else:
                # ── 2. Fallback: EfficientNet-B0 ──────────────────────────
                input_tensor = self.preprocess(crop_bgr)
                _, _, logits = self.gradcam.generate_heatmap(input_tensor, target_class=1)
                probs      = torch.softmax(logits, dim=1).squeeze().cpu().detach().numpy()
                real_prob  = float(probs[0])
                fake_prob  = float(probs[1])
                prediction = "AI-GENERATED" if fake_prob >= 0.5 else "REAL"
                conf       = fake_prob if prediction != "REAL" else real_prob
                confidence_pct = round(conf * 100, 1)

            # ── 3. Grad-CAM heatmap (visual explanation) ──────────────────
            input_tensor = self.preprocess(crop_bgr)
            heatmap, _, _ = self.gradcam.generate_heatmap(input_tensor, target_class=1)
            heatmap_resized, colormap_bgr, overlay_bgr = create_heatmap_overlay(
                crop_bgr, heatmap, alpha=0.45
            )

            # ── 4. Spatial region analysis ────────────────────────────────
            top_regions, region_texts, dominant_region = analyze_activation_regions(heatmap_resized)

            # ── 5. FFT frequency analysis ─────────────────────────────────
            freq_score, freq_evidence, fft_spectrum_bgr = analyze_frequency_domain(crop_bgr)

            # ── 6. Build evidence list ────────────────────────────────────
            evidence_reasons = self._build_evidence(
                hf_result, region_texts, freq_score, freq_evidence,
                prediction, confidence_pct, image_type
            )

            # ── 7. Save visuals ───────────────────────────────────────────
            uid = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_img{idx}"
            crop_path    = os.path.join(self.output_dir, f"{uid}_crop.jpg")
            heatmap_path = os.path.join(self.output_dir, f"{uid}_heatmap.jpg")
            overlay_path = os.path.join(self.output_dir, f"{uid}_overlay.jpg")
            fft_path     = os.path.join(self.output_dir, f"{uid}_fft.jpg")

            cv2.imwrite(crop_path, crop_bgr)
            cv2.imwrite(heatmap_path, colormap_bgr)
            cv2.imwrite(overlay_path, overlay_bgr)
            if fft_spectrum_bgr is not None:
                cv2.imwrite(fft_path, fft_spectrum_bgr)

            results.append({
                'face_id':                  idx + 1,
                'bounding_box':             box,
                'face_detection_confidence': round(face_conf * 100, 1),
                'is_fallback_face':         not has_human_face,
                'image_type':               image_type,
                'prediction':               prediction,
                'confidence':               confidence_pct,
                'real_probability':         round(real_prob * 100, 1),
                'fake_probability':         round(fake_prob * 100, 1),
                'explanation_reasons':      evidence_reasons,
                'dominant_region':          dominant_region,
                'activated_regions':        top_regions,
                'frequency_anomaly_score':  round(freq_score * 100, 1),
                'model_used':               hf_result.get("model_used", "EfficientNet-B0"),
                'heatmap_urls': {
                    'crop':    f"/outputs/heatmaps/{os.path.basename(crop_path)}",
                    'heatmap': f"/outputs/heatmaps/{os.path.basename(heatmap_path)}",
                    'overlay': f"/outputs/heatmaps/{os.path.basename(overlay_path)}",
                    'fft':     f"/outputs/heatmaps/{os.path.basename(fft_path)}" if fft_spectrum_bgr is not None else None,
                }
            })

        return results
