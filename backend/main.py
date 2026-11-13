import os
import sys

# Ensure project root is in sys.path so running directly from VS Code works out-of-the-box
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes import router
from backend.database.db import init_db
from backend.services.detector_service import get_detector_engine

app = FastAPI(
    title="Deepfake Detection and Explainable AI System",
    description="FastAPI Backend for PyTorch EfficientNet-B0 Deepfake Detector & Grad-CAM visual explanations.",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs('outputs/heatmaps', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('frontend', exist_ok=True)

# Mount static directories
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Include API Router
app.include_router(router)

@app.on_event("startup")
def startup_event():
    """
    On server boot: initialize database schema & load ML inference engine.
    """
    init_db()
    get_detector_engine()

@app.get("/")
def read_root():
    """
    Serve the frontend dashboard homepage.
    """
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Deepfake Detection API is running. Frontend index.html not found."}

if __name__ == '__main__':
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1")
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload)
