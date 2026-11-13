import os
import cv2
import numpy as np

def generate_synthetic_face_image(is_fake=False, size=(256, 256)):
    """
    Generate synthetic placeholder face-like test image.
    Real: smooth gradients, natural noise.
    Fake: local frequency distortion, subtle box blur or checkerboard overlay.
    """
    h, w = size
    image = np.ones((h, w, 3), dtype=np.uint8) * 180
    
    # Face oval background
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.45))
    cv2.ellipse(image, center, axes, 0, 0, 360, (210, 180, 160), -1)
    
    # Eyes
    left_eye = (int(w * 0.38), int(h * 0.4))
    right_eye = (int(w * 0.62), int(h * 0.4))
    cv2.circle(image, left_eye, int(w * 0.06), (255, 255, 255), -1)
    cv2.circle(image, right_eye, int(w * 0.06), (255, 255, 255), -1)
    cv2.circle(image, left_eye, int(w * 0.025), (40, 30, 20), -1)
    cv2.circle(image, right_eye, int(w * 0.025), (40, 30, 20), -1)
    
    # Mouth
    mouth_center = (int(w * 0.5), int(h * 0.72))
    cv2.ellipse(image, mouth_center, (int(w * 0.15), int(h * 0.06)), 0, 0, 180, (60, 40, 140), -1)
    
    # Add subtle skin texture noise
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.float32)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    if is_fake:
        # Inject artificial artifacts (e.g. sharp localized grid noise around eyes or boundary)
        eye_patch = image[int(h*0.35):int(h*0.48), int(w*0.30):int(w*0.70)]
        grid_pattern = ((np.arange(eye_patch.shape[0])[:, None] + np.arange(eye_patch.shape[1])) % 2 == 0) * 45
        image[int(h*0.35):int(h*0.48), int(w*0.30):int(w*0.70), 0] = np.clip(
            eye_patch[:, :, 0].astype(np.int16) + grid_pattern, 0, 255
        ).astype(np.uint8)
        
        # Facial boundary blending artifact
        cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 255), 3)
        
    return image

def create_sample_dataset(root_dir='dataset', num_per_class=10):
    """
    Populate dataset/ train/ val/ test directories with synthetic face samples.
    """
    splits = {
        'train': num_per_class,
        'validation': max(3, num_per_class // 3),
        'test': max(3, num_per_class // 3)
    }
    
    for split, count in splits.items():
        for category in ['real', 'fake']:
            target_dir = os.path.join(root_dir, split, category)
            os.makedirs(target_dir, exist_ok=True)
            is_fake = (category == 'fake')
            
            for i in range(1, count + 1):
                img = generate_synthetic_face_image(is_fake=is_fake)
                filename = f"{category}_{i:03d}.jpg"
                filepath = os.path.join(target_dir, filename)
                cv2.imwrite(filepath, img)
                
    # Also save a couple of sample images in data/sample_images/ for quick UI analysis
    sample_dir = os.path.join('data', 'sample_images')
    os.makedirs(sample_dir, exist_ok=True)
    cv2.imwrite(os.path.join(sample_dir, 'sample_real.jpg'), generate_synthetic_face_image(is_fake=False))
    cv2.imwrite(os.path.join(sample_dir, 'sample_deepfake.jpg'), generate_synthetic_face_image(is_fake=True))
    
    print(f"[Dataset] Created sample dataset structure in '{root_dir}' and '{sample_dir}'")

if __name__ == '__main__':
    create_sample_dataset()
