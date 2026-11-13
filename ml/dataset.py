import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(img_size=224):
    """
    Albumentations augmentation pipeline for training dataset.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.ImageCompression(quality_range=(70, 100), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    """
    Validation / Testing transforms without random augmentations.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset loading images from real/ and fake/ subdirectories.
    Structure:
    dataset_dir/
      ├── real/ (*.jpg, *.png, *.webp) -> 0
      └── fake/ (*.jpg, *.png, *.webp) -> 1
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.targets = []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        if os.path.exists(real_dir):
            for file in os.listdir(real_dir):
                if file.lower().endswith(valid_extensions):
                    self.samples.append(os.path.join(real_dir, file))
                    self.targets.append(0) # 0 = REAL
                    
        if os.path.exists(fake_dir):
            for file in os.listdir(fake_dir):
                if file.lower().endswith(valid_extensions):
                    self.samples.append(os.path.join(fake_dir, file))
                    self.targets.append(1) # 1 = DEEPFAKE
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image file: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, target

def get_class_weighted_sampler(dataset):
    """
    Creates a WeightedRandomSampler to handle class imbalance in training dataset.
    """
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)
    
    if len(class_counts) < 2 or torch.any(class_counts == 0):
        return None
        
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
