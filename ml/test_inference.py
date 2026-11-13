import os
import cv2
from ml.create_sample_dataset import create_sample_dataset
from ml.inference import DeepfakeInferenceEngine

def main():
    print("[Test] Creating sample dataset...")
    create_sample_dataset(root_dir='dataset', num_per_class=5)
    
    print("[Test] Initializing Inference Engine...")
    engine = DeepfakeInferenceEngine()
    
    sample_path = os.path.join('data', 'sample_images', 'sample_deepfake.jpg')
    if not os.path.exists(sample_path):
        print(f"Sample image not found at {sample_path}")
        return
        
    image_bgr = cv2.imread(sample_path)
    print(f"[Test] Analyzing sample image {sample_path} ({image_bgr.shape})...")
    
    results = engine.analyze_image(image_bgr)
    print(f"[Test] Analysis finished. Detected {len(results)} face(s).")
    for r in results:
        print("--- Result Summary ---")
        print(f"Face ID: {r['face_id']}")
        print(f"Prediction: {r['prediction']} ({r['confidence']}%)")
        print(f"Dominant Region: {r['dominant_region']}")
        print(f"Frequency Anomaly Score: {r['frequency_anomaly_score']}%")
        print("Reasons:")
        for reason in r['explanation_reasons']:
            print(f"  - {reason}")
        print("Heatmap URLs:", r['heatmap_urls'])

if __name__ == '__main__':
    main()
