import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

# sklearn is imported lazily inside functions to avoid DLL-load issues at import time

def compute_metrics(y_true, y_pred_probs):
    """
    Computes Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
    
    y_true: 1D array of ground truth labels (0 = REAL, 1 = DEEPFAKE)
    y_pred_probs: 1D array or 2D array of class probabilities for DEEPFAKE (index 1)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    y_true = np.array(y_true)
    if y_pred_probs.ndim == 2:
        probs = y_pred_probs[:, 1]
    else:
        probs = y_pred_probs
        
    y_pred = (probs >= 0.5).astype(int)
    
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except Exception:
        roc_auc = 0.5 # Fallback if single class present in batch
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def save_evaluation_plots(train_losses, val_losses, y_true, y_probs, output_dir='outputs/reports'):
    """
    Generates and saves Training/Validation Loss graph, ROC Curve, and Confusion Matrix heatmap.
    """
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Training & Validation Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='#3b82f6', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='#ef4444', linewidth=2, linestyle='--')
    plt.title('Training & Validation Loss Curve', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_path, dpi=150)
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    if len(y_true) > 0 and len(y_probs) > 0:
        if y_probs.ndim == 2:
            probs = y_probs[:, 1]
        else:
            probs = y_probs
        y_pred = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=12)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['REAL', 'DEEPFAKE'], fontsize=10)
        plt.yticks(tick_marks, ['REAL', 'DEEPFAKE'], fontsize=10)
        
        thresh = cm.max() / 2. if cm.max() > 0 else 1.
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontsize=12, fontweight='bold')
                         
        plt.ylabel('True Label', fontsize=11)
        plt.xlabel('Predicted Label', fontsize=11)
        plt.tight_layout()
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        plt.close()
        
        # 3. ROC Curve Plot
        try:
            fpr, tpr, _ = roc_curve(y_true, probs)
            auc_val = roc_auc_score(y_true, probs)
            
            plt.figure(figsize=(7, 5))
            plt.plot(fpr, tpr, color='#10b981', linewidth=2, label=f'ROC Curve (AUC = {auc_val:.3f})')
            plt.plot([0, 1], [0, 1], color='#9ca3af', linestyle='--')
            plt.title('Receiver Operating Characteristic (ROC)', fontsize=14, fontweight='bold', pad=12)
            plt.xlabel('False Positive Rate', fontsize=11)
            plt.ylabel('True Positive Rate', fontsize=11)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(frameon=True)
            plt.tight_layout()
            roc_path = os.path.join(output_dir, 'roc_curve.png')
            plt.savefig(roc_path, dpi=150)
            plt.close()
        except Exception:
            pass
            
    print(f"[Evaluation] Saved evaluation plots to {output_dir}")
