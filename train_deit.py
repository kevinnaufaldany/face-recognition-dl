import os
import time
import warnings
import json
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Filter sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from datareader import create_dataloaders
from model_deit import create_model


def plot_metrics(history, save_dir):
    """
    Plot metrics setelah training selesai.
    Buat plot terpisah untuk loss, accuracy, precision, recall, dan F1.
    
    Args:
        history: Dict dengan training history
        save_dir: Directory untuk save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=150)
    plt.close()
    
    # 2. Plot Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=150)
    plt.close()
    
    # 3. Plot F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2, marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_score.png'), dpi=150)
    plt.close()
    
    # 4. Plot Precision and Recall
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_precision'], 'g-', label='Train Precision', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, history['val_precision'], 'purple', label='Val Precision', linewidth=2, marker='s', markersize=4)
    plt.plot(epochs, history['train_recall'], 'b--', label='Train Recall', linewidth=2, marker='D', markersize=4)
    plt.plot(epochs, history['val_recall'], 'r--', label='Val Recall', linewidth=2, marker='^', markersize=4)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Precision and Recall', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_recall.png'), dpi=150)
    plt.close()
    
    print(f"\n  Visualizations saved to {save_dir}")
    print(f"    - loss.png")
    print(f"    - accuracy.png")
    print(f"    - f1_score.png")
    print(f"    - precision_recall.png")


def plot_confusion_matrix(y_true, y_pred, num_classes, save_dir):
    """
    Plot confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
        save_dir: Directory untuk save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues', 
                cbar_kws={'label': 'Normalized Frequency'},
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    # Plot raw counts confusion matrix
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Greens',
                cbar_kws={'label': 'Count'},
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_counts.png'), dpi=150)
    plt.close()
    
    print(f"    - confusion_matrix.png")
    print(f"    - confusion_matrix_counts.png")


class MetricsTracker:
    """Class untuk tracking metrics selama training"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.loss_sum = 0.0
        self.count = 0
    
    def update(self, preds, targets, loss):
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.loss_sum += loss * len(targets)
        self.count += len(targets)
    
    def compute(self):
        avg_loss = self.loss_sum / self.count if self.count > 0 else 0.0
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class EarlyStopping:
    """Early stopping untuk menghentikan training jika tidak ada improvement"""
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
        """
        Args:
            patience (int): Berapa epoch menunggu sebelum stop
            min_delta (float): Minimum perubahan untuk dianggap improvement
            mode (str): 'max' untuk metric yang lebih besar lebih baik (accuracy),
                       'min' untuk metric yang lebih kecil lebih baik (loss)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Training untuk satu epoch"""
    model.train()
    metrics = MetricsTracker()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]', ncols=120)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training only if CUDA available
        if scaler is not None:
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Backward pass dengan gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Update metrics
        _, preds = torch.max(outputs, 1)
        metrics.update(preds, targets, loss.item())
        
        # Update progress bar
        current_metrics = metrics.compute()
        pbar.set_postfix({
            'loss': f'{current_metrics["loss"]:.4f}',
            'acc': f'{current_metrics["accuracy"]:.4f}'
        })
    
    return metrics.compute()


@torch.no_grad()
def validate(model, val_loader, criterion, device, desc='VAL', return_predictions=False):
    """Validation/Testing"""
    model.eval()
    metrics = MetricsTracker()
    
    all_preds = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'[{desc}]', ncols=120, leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Mixed precision inference only if CUDA available
        if torch.cuda.is_available():
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
        
        _, preds = torch.max(outputs, 1)
        metrics.update(preds, targets, loss.item())
        
        if return_predictions:
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        current_metrics = metrics.compute()
        pbar.set_postfix({
            'loss': f'{current_metrics["loss"]:.4f}',
            'acc': f'{current_metrics["accuracy"]:.4f}'
        })
    
    if return_predictions:
        return metrics.compute(), all_preds, all_targets
    return metrics.compute()


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def train(
    data_dir='dataset/Train',
    num_classes=70,
    batch_size=8,
    epochs=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    device=None,
    save_dir='checkpoints'
):
    """
    Main training function untuk DeiT-Small
    
    Args:
        data_dir (str): Path ke folder Train
        num_classes (int): Jumlah kelas
        batch_size (int): Batch size
        epochs (int): Jumlah epochs
        learning_rate (float): Learning rate untuk optimizer
        weight_decay (float): Weight decay untuk AdamW
        device (str): Device untuk training
        save_dir (str): Directory untuk menyimpan checkpoints
    """
    # Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verify CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, using CPU")
    
    print("="*80)
    print("DeiT-SMALL MODEL TRAINING")
    print("="*80)
    print(f"Device: {device} | Batch: {batch_size} | Epochs: {epochs} | LR: {learning_rate}")
    print(f"Input Size: 512x512 | Embedding: 128 | Architecture: DeiT-Small")
    print(f"Dropout: 0.3 | Label Smoothing: 0.1 | Stochastic Depth: 0.1")
    print("="*80)
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"deit_small_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir, 
        batch_size=batch_size,
        num_workers=4  # Use 4 workers for faster data loading
    )
    
    # Create model
    print("\nMODEL ARCHITECTURE")
    print("-" * 90)
    model = create_model(num_classes=num_classes, pretrained=True, dropout=0.3, device=device)
    
    total_params = model.get_num_total_params()
    trainable_params = model.get_num_trainable_params()
    non_trainable = total_params - trainable_params
    
    print(f"  Architecture  : DeiT-Small (Distilled)")
    print(f"  Pretrained    : Facebook/ImageNet-1K")
    print(f"  Input size    : 512x512x3")
    print(f"  Embedding size: 128 (reduced from 384)")
    print(f"  Classes       : {num_classes}")
    print(f"  Total params  : {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable     : {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-trainable : {non_trainable:,} ({non_trainable/1e6:.2f}M)")
    print(f"  Device        : {str(device).upper()}")
    
    # Setup training components
    print("\nTRAINING CONFIGURATION")
    print("-" * 90)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Warmup + Cosine scheduler
    from torch.optim.lr_scheduler import LambdaLR
    warmup_epochs = 5
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Only use GradScaler if CUDA is available
    if torch.cuda.is_available():
        scaler = GradScaler()
    else:
        scaler = None
        
    early_stopping = EarlyStopping(patience=7, mode='max')
    
    print(f"  Optimizer     : AdamW (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"  Scheduler     : Cosine + Warmup ({warmup_epochs} epochs)")
    print(f"  Loss function : CrossEntropyLoss (label_smoothing=0.1)")
    print(f"  Batch size    : {batch_size}")
    print(f"  Epochs        : {epochs}")
    print(f"  Dropout       : 0.3")
    print(f"  Augmentation  : ColorJitter + RandomErasing(p=0.2) + HFlip(p=0.5)")
    print(f"  Mixed precision: {'Enabled (AMP)' if scaler is not None else 'Disabled (CPU)'}")
    print(f"  Grad clipping : max_norm=1.0")
    print(f"  Early stopping: patience={early_stopping.patience}")
    
    # Training loop
    print("\n" + "="*90)
    print("TRAINING START")
    print("="*90)
    
    best_val_acc = 0.0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_precision': [], 'train_recall': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_precision': [], 'val_recall': [],
        'lr': []
    }
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, desc='VAL')
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Print metrics summary
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Acc: {train_metrics['accuracy']*100:.2f}% | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}%")
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['train_f1'].append(train_metrics['f1'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['lr'].append(current_lr)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            # Delete previous best checkpoint
            if best_epoch > 0:
                old_checkpoint = run_dir / f'best_epoch{best_epoch}.pth'
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
            
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            best_model_path = run_dir / f'best_epoch{epoch}.pth'
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_model_path)
            print(f"✓ Best model saved: {best_model_path.name} (Acc: {best_val_acc*100:.2f}%)")
        
        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            print(f"No improvement for {early_stopping.patience} epochs")
            break
    
    total_time = time.time() - start_time
    
    # Load best model for final validation
    checkpoint = torch.load(run_dir / f'best_epoch{best_epoch}.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation with confusion matrix
    print("\nGenerating confusion matrix...")
    _, y_pred, y_true = validate(model, val_loader, criterion, device, desc='FINAL', return_predictions=True)
    
    # Save training history as JSON
    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc
    history['total_epochs'] = len(history['train_loss'])
    
    history_path = run_dir / 'history_train.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating metric visualizations...")
    plot_metrics(history, str(run_dir))
    plot_confusion_matrix(y_true, y_pred, num_classes, str(run_dir))
    
    # Print summary
    print("\n" + "="*90)
    print("TRAINING COMPLETED")
    print("="*90)
    print(f"  Total time        : {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"  Best epoch        : {best_epoch}/{epochs}")
    print(f"  Best val accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Best val F1 score : {history['val_f1'][best_epoch-1]:.4f}")
    print(f"  Final train acc   : {history['train_acc'][-1]:.4f} ({history['train_acc'][-1]*100:.2f}%)")
    print(f"  Final train loss  : {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss    : {history['val_loss'][-1]:.4f}")
    print(f"\n  Checkpoints saved : {run_dir}")
    print(f"    - best_epoch{best_epoch}.pth")
    print(f"    - history_train.json")
    print("="*90)
    
    return model, history


if __name__ == "__main__":
    # Training configuration
    CONFIG = {
        'data_dir': 'dataset/Train',
        'num_classes': 70,
        'batch_size': 4,
        'epochs': 50,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'save_dir': 'checkpoints'
    }
    
    # Start training
    model, history = train(**CONFIG)
