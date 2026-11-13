import os
import argparse
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from ml.model import build_model, save_checkpoint
from ml.dataset import (
    DeepfakeDataset,
    get_train_transforms,
    get_val_transforms,
    get_class_weighted_sampler
)
from ml.evaluation import compute_metrics, save_evaluation_plots

def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Deepfake Detector")
    parser.add_argument('--dataset_dir', type=str, default='dataset', help="Root dataset directory")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--img_size', type=int, default=224, help="Input image size")
    parser.add_argument('--checkpoint_path', type=str, default='models/deepfake_detector.pth', help="Path to save best model")
    parser.add_argument('--reports_dir', type=str, default='outputs/reports', help="Directory to save evaluation reports")
    parser.add_argument('--patience', type=int, default=4, help="Early stopping patience")
    return parser.parse_args()

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        
        all_targets.extend(targets.cpu().numpy())
        all_probs.append(probs)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_targets, all_probs)
    return epoch_loss, metrics

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            
            all_targets.extend(targets.cpu().numpy())
            all_probs.append(probs)
            
    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.vstack(all_probs) if len(all_probs) > 0 else np.array([])
    metrics = compute_metrics(all_targets, all_probs)
    return epoch_loss, metrics, all_targets, all_probs

def run_training(dataset_dir='dataset', epochs=10, batch_size=16, lr=1e-4,
                 img_size=224, checkpoint_path='models/deepfake_detector.pth',
                 reports_dir='outputs/reports', patience=4):
                 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Training] Using device: {device}")
    
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'validation')
    test_dir = os.path.join(dataset_dir, 'test')
    
    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        raise FileNotFoundError(f"Training directory '{train_dir}' is empty or missing! Please add train/real and train/fake data.")
        
    # Load Datasets
    train_dataset = DeepfakeDataset(train_dir, transform=get_train_transforms(img_size))
    val_dataset = DeepfakeDataset(val_dir, transform=get_val_transforms(img_size)) if os.path.exists(val_dir) else train_dataset
    test_dataset = DeepfakeDataset(test_dir, transform=get_val_transforms(img_size)) if os.path.exists(test_dir) else None
    
    print(f"[Training] Samples - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset) if test_dataset else 0}")
    
    sampler = get_class_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler if sampler else None,
        shuffle=True if not sampler else False,
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = build_model(num_classes=2, pretrained=True, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    best_targets, best_probs = [], []
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, targets, probs = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']*100:.1f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.1f}%, "
              f"F1: {val_metrics['f1_score']:.3f}, AUC: {val_metrics['roc_auc']:.3f}")
              
        # Early stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_targets, best_probs = targets, probs
            
            save_checkpoint(model, checkpoint_path, config={
                'architecture': 'efficientnet_b0',
                'num_classes': 2,
                'class_names': ['REAL', 'DEEPFAKE'],
                'image_size': img_size,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'val_metrics': val_metrics,
                'trained_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Training] Early stopping triggered after {epoch} epochs.")
                break
                
    elapsed = time.time() - start_time
    print(f"[Training] Finished in {elapsed/60:.2f} minutes.")
    
    # Save evaluation graphs and reports
    save_evaluation_plots(train_losses, val_losses, best_targets, best_probs, output_dir=reports_dir)
    
    # Final Test Set Evaluation
    if test_dataset and len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_loss, test_metrics, _, _ = validate(model, test_loader, criterion, device)
        print(f"[Testing] Final Test Loss: {test_loss:.4f} | Acc: {test_metrics['accuracy']*100:.1f}%, F1: {test_metrics['f1_score']:.3f}, AUC: {test_metrics['roc_auc']:.3f}")
        with open(os.path.join(reports_dir, 'test_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)

if __name__ == '__main__':
    args = parse_args()
    run_training(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        checkpoint_path=args.checkpoint_path,
        reports_dir=args.reports_dir,
        patience=args.patience
    )
