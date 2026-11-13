from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str
    model_checkpoint_loaded: bool
    device: str
    version: str = "1.0.0"

class HeatmapURLs(BaseModel):
    crop: str
    heatmap: str
    overlay: str
    fft: Optional[str] = None

class ActivatedRegion(BaseModel):
    region: str
    score: float
    severity: str
    description: str

class FaceAnalysisResult(BaseModel):
    face_id: int
    bounding_box: List[int]
    face_detection_confidence: float
    is_fallback_face: bool
    prediction: str # REAL, DEEPFAKE, AI-GENERATED
    confidence: float # Percentage
    real_probability: float
    fake_probability: float
    explanation_reasons: List[str]
    dominant_region: str
    activated_regions: List[ActivatedRegion]
    frequency_anomaly_score: float
    heatmap_urls: HeatmapURLs
    image_type: Optional[str] = None
    model_used: Optional[str] = None

class PredictResponse(BaseModel):
    filename: str
    total_faces_detected: int
    faces: List[FaceAnalysisResult]
    primary_result: FaceAnalysisResult
    timestamp: str

class HistoryItem(BaseModel):
    id: int
    filename: str
    prediction: str
    confidence: float
    face_count: int
    dominant_region: Optional[str] = None
    frequency_score: Optional[float] = None
    explanation_reasons: List[str]
    heatmap_path: Optional[str] = None
    created_at: str

class TrainResponse(BaseModel):
    status: str
    message: str
