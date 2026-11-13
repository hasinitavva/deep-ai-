import time
import cv2
import numpy as np
from fastapi import HTTPException
from ml.inference import DeepfakeInferenceEngine
from backend.database.repository import save_prediction

_detector_engine = None

def get_detector_engine():
    global _detector_engine
    if _detector_engine is None:
        _detector_engine = DeepfakeInferenceEngine()
    return _detector_engine

def process_image_upload(image_bytes: bytes, filename: str):
    """
    Decodes uploaded image bytes, runs face detection & EfficientNet inference,
    saves record to SQLite history DB, and returns formatted prediction result.
    """
    # Convert image bytes to OpenCV BGR matrix
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if image_bgr is None or image_bgr.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image file or unsupported image format.")
        
    engine = get_detector_engine()
    faces_results = engine.analyze_image(image_bgr)
    
    if not faces_results:
        raise HTTPException(status_code=422, detail="Unable to process face features in the provided image.")
        
    # Primary result is the first detected face or highest confidence prediction
    primary_result = faces_results[0]
    for face_res in faces_results:
        if face_res['prediction'] in ('DEEPFAKE', 'AI-GENERATED'):
            primary_result = face_res
            break
            
    # Save to SQLite database
    try:
        save_prediction(filename, primary_result, total_faces=len(faces_results))
    except Exception as e:
        print(f"[DetectorService] DB save warning: {e}")
        
    return {
        'filename': filename,
        'total_faces_detected': len(faces_results),
        'faces': faces_results,
        'primary_result': primary_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
