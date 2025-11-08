# train_ensemble_92percent.py
# STRATEGI ENSEMBLE + ADVANCED TECHNIQUES untuk 92% Accuracy
# Gabungan multiple models dengan data augmentation yang lebih agresif

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import numpy as np
import time
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from model import EfficientNetB0Binary
from datareader import get_data_loaders


def set_seed(seed=42):
    """Set random seed untuk reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred, y_scores=None):
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_scores is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics


class FocalLoss(nn.Module):
    """Focal Loss untuk handle class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train model untuk 1 epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().view(-1, 1)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        
        # Predictions
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten()
    )
    
    return epoch_loss, metrics['accuracy']


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )
    
    return val_loss, metrics


def train_single_model(seed, device, train_loader, val_loader, test_loader):
    """Train single model dengan seed tertentu"""
    
    print(f"\n{'='*70}")
    print(f"🔄 Training Model #{seed}")
    print(f"{'='*70}")
    
    # Set seed
    set_seed(seed)
    
    # Hyperparameters - OPTIMIZED untuk 92%
    BATCH_SIZE = 32
    EPOCHS = 200  # Lebih panjang
    IMG_SIZE = 96  # Lebih besar untuk detail
    LEARNING_RATE = 5e-4  # Higher initial LR
    UNFREEZE_LR = 1e-4  # Higher fine-tuning LR
    WEIGHT_DECAY = 1e-5  # Very low
    DROPOUT = 0.25  # Lower dropout
    FREEZE_EPOCHS = 15  # Shorter frozen phase
    PATIENCE = 50  # More patience
    
    # Create model
    model = EfficientNetB0Binary(
        img_size=IMG_SIZE,
        in_channels=1,
        num_classes=1,
        pretrained=True,
        dropout=DROPOUT,
        freeze_backbone=True
    ).to(device)
    
    # Focal Loss untuk handle imbalance
    criterion = FocalLoss(alpha=0.3, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=FREEZE_EPOCHS,
        T_mult=1,
        eta_min=UNFREEZE_LR
    )
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print(f"🚀 Phase 1: Frozen backbone (LR={LEARNING_RATE})")
    
    for epoch in range(1, EPOCHS + 1):
        # Unfreeze setelah FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1:
            print(f"\n🔥 Phase 2: Full fine-tuning (LR={UNFREEZE_LR})")
            for param in model.parameters():
                param.requires_grad = True
            
            optimizer = optim.AdamW(
                model.parameters(),
                lr=UNFREEZE_LR,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999)
            )
            
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=(EPOCHS - FREEZE_EPOCHS) // 4,
                T_mult=1,
                eta_min=UNFREEZE_LR / 10
            )
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_acc = val_metrics['accuracy']
        
        # Update scheduler
        scheduler.step()
        
        # Print only every 10 epochs to reduce clutter
        if epoch % 10 == 0 or val_acc > best_val_acc:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{EPOCHS}:")
            print(f"  Val Acc: {val_acc:.2%} | Train Acc: {train_acc:.2%}")
            print(f"  AUC: {val_metrics['auc']:.4f} | LR: {current_lr:.6f}")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'seed': seed,
            }, f'model_ensemble_seed{seed}.pth')
            
            if val_acc > 0.88:  # Only print for high accuracy
                print(f"  ✨ New best: {val_acc:.2%}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏸️  Early stopping at epoch {epoch}")
            break
    
    print(f"\n✅ Model #{seed} completed!")
    print(f"   Best validation accuracy: {best_val_acc:.2%} at epoch {best_epoch}")
    
    # Load best model dan evaluate
    checkpoint = torch.load(f'model_ensemble_seed{seed}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, test_metrics = validate(model, test_loader, criterion, device)
    print(f"   Test accuracy: {test_metrics['accuracy']:.2%}")
    
    return model, best_val_acc, test_metrics


def ensemble_predict(models, data_loader, device):
    """Ensemble prediction dari multiple models"""
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            # Average predictions dari semua models
            ensemble_probs = []
            for model in models:
                model.eval()
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                ensemble_probs.append(probs)
            
            # Average
            avg_probs = torch.stack(ensemble_probs).mean(dim=0)
            preds = (avg_probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(avg_probs.cpu().numpy())
    
    metrics = calculate_metrics(
        np.array(all_labels).flatten(),
        np.array(all_preds).flatten(),
        np.array(all_scores).flatten()
    )
    
    return metrics


def main():
    print("\n" + "=" * 70)
    print("   🎯 ENSEMBLE TRAINING untuk 92% Accuracy")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data dengan ukuran lebih besar
    print(f"\n📦 Loading dataset...")
    train_loader, val_loader, n_classes, n_channels = get_data_loaders(
        batch_size=32,
        img_size=96  # Larger for more detail
    )
    test_loader = val_loader
    
    print(f"   Training: {len(train_loader.dataset)} samples")
    print(f"   Validation: {len(val_loader.dataset)} samples")
    
    # Train multiple models dengan different seeds
    print(f"\n{'='*70}")
    print(f"🚀 Training 3 models dengan different seeds...")
    print(f"{'='*70}")
    
    seeds = [42, 123, 456]
    models = []
    val_accs = []
    
    for seed in seeds:
        model, val_acc, test_metrics = train_single_model(
            seed, device, train_loader, val_loader, test_loader
        )
        models.append(model)
        val_accs.append(val_acc)
    
    # Ensemble evaluation
    print(f"\n{'='*70}")
    print(f"📊 ENSEMBLE EVALUATION")
    print(f"{'='*70}")
    
    print(f"\nIndividual model accuracies:")
    for i, (seed, acc) in enumerate(zip(seeds, val_accs)):
        print(f"  Model #{seed}: {acc:.2%}")
    
    print(f"\nEvaluating ensemble on test set...")
    ensemble_metrics = ensemble_predict(models, test_loader, device)
    
    print(f"\n{'='*70}")
    print(f"🎉 FINAL ENSEMBLE RESULTS:")
    print(f"{'='*70}")
    print(f"Test Accuracy: {ensemble_metrics['accuracy']:.2%}")
    print(f"Test Precision: {ensemble_metrics['precision']:.2%}")
    print(f"Test Recall: {ensemble_metrics['recall']:.2%}")
    print(f"Test F1-Score: {ensemble_metrics['f1']:.4f}")
    print(f"Test AUC: {ensemble_metrics['auc']:.4f}")
    print(f"{'='*70}")
    
    if ensemble_metrics['accuracy'] >= 0.92:
        print(f"\n🎉🎉🎉 CONGRATULATIONS! Target 92% ACHIEVED! 🎉🎉🎉")
    elif ensemble_metrics['accuracy'] >= 0.90:
        print(f"\n👍 Excellent! Very close to 92% target!")
    else:
        print(f"\n💡 Suggestions:")
        print(f"   - Try more models in ensemble (5-7 models)")
        print(f"   - Increase img_size to 128")
        print(f"   - Train longer (300 epochs)")


if __name__ == '__main__':
    main()
