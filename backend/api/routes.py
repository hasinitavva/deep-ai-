import os
import threading
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from typing import List

from backend.api.schemas import (
    HealthResponse, PredictResponse, HistoryItem, TrainResponse
)
from backend.services.detector_service import process_image_upload, get_detector_engine
from backend.database.repository import fetch_history
from ml.train import run_training

router = APIRouter()

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024 # 15MB

@router.get('/health', response_model=HealthResponse)
def health_check():
    """
    Check backend status, GPU availability, and model loading status.
    """
    engine = get_detector_engine()
    return HealthResponse(
        status="healthy",
        model_checkpoint_loaded=engine.is_checkpoint_loaded,
        device=str(engine.device),
        version="1.0.0"
    )

@router.post('/predict', response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Analyze image for Deepfake detection & Grad-CAM visual explanation.
    """
    # 1. Validate File Extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )
        
    # 2. Read File Bytes and Check Size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds 15MB limit ({len(contents) / (1024*1024):.1f}MB)."
        )
        
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    # 3. Process Image through Detector Service
    result = process_image_upload(contents, file.filename)
    return result

@router.get('/history', response_model=List[HistoryItem])
def get_prediction_history(limit: int = 50):
    """
    Retrieve past prediction history records from SQLite database.
    """
    return fetch_history(limit=limit)

def _background_training_task():
    try:
        run_training(epochs=5, batch_size=16)
    except Exception as e:
        print(f"[BackgroundTraining] Error: {e}")

@router.post('/train', response_model=TrainResponse)
def trigger_training(background_tasks: BackgroundTasks):
    """
    Trigger PyTorch model training in the background using available dataset/ directory.
    """
    train_dir = os.path.join('dataset', 'train')
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        return TrainResponse(
            status="error",
            message="Dataset directory 'dataset/train' is empty. Run 'python ml/create_sample_dataset.py' or add images to dataset/ train/real and train/fake."
        )
        
    background_tasks.add_task(_background_training_task)
    return TrainResponse(
        status="started",
        message="Model training pipeline launched in the background. Check logs or models/ for output."
    )
