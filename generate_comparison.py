"""
Generate comparison plots between ConvNeXt-Tiny and Swin V2 Tiny
Reads training history and creates side-by-side comparison visualizations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Load training histories
convnext_dir = Path('checkpoints\convnext_tiny_20251201_144518')
swin_dir = Path('checkpoints\swin_v2_tiny_20251201_150017')

with open(convnext_dir / 'history_train.json', 'r') as f:
    convnext_history = json.load(f)

with open(swin_dir / 'history_train.json', 'r') as f:
    swin_history = json.load(f)

# Create output directory
output_dir = Path('model_comparison')
output_dir.mkdir(exist_ok=True)

# >>>---------------------------------------------
# >>> LR FIX PATCH: NORMALIZE LR LENGTH
# >>>---------------------------------------------
def normalize_lr(lr_list, target_len):
    if lr_list is None:
        return None
    if len(lr_list) == target_len:
        return lr_list
    if len(lr_list) == 1:
        return [lr_list[0]] * target_len
    if len(lr_list) > target_len:  # per-step LR
        step = len(lr_list) // target_len
        return [lr_list[(i+1)*step - 1] for i in range(target_len)]
    return lr_list


# Function to create comparison plots
def plot_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Training Loss Comparison
    ax = axes[0, 0]
    epochs_convnext = range(1, len(convnext_history['train_loss']) + 1)
    epochs_swin = range(1, len(swin_history['train_loss']) + 1)
    
    ax.plot(epochs_convnext, convnext_history['train_loss'], 'b-', 
            label='ConvNeXt Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs_swin, swin_history['train_loss'], 'r-', 
            label='Swin V2 Train', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation Loss Comparison
    ax = axes[0, 1]
    ax.plot(epochs_convnext, convnext_history['val_loss'], 'b-', 
            label='ConvNeXt Val', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs_swin, swin_history['val_loss'], 'r-', 
            label='Swin V2 Val', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Training Accuracy Comparison
    ax = axes[0, 2]
    ax.plot(epochs_convnext, [x*100 for x in convnext_history['train_acc']], 'b-', 
            label='ConvNeXt Train', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs_swin, [x*100 for x in swin_history['train_acc']], 'r-', 
            label='Swin V2 Train', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 4. Validation Accuracy Comparison (MAIN METRIC)
    ax = axes[1, 0]
    ax.plot(epochs_convnext, [x*100 for x in convnext_history['val_acc']], 'b-', 
            label='ConvNeXt Val', linewidth=3, marker='o', markersize=6)
    ax.plot(epochs_swin, [x*100 for x in swin_history['val_acc']], 'r-', 
            label='Swin V2 Val', linewidth=3, marker='s', markersize=6)
    
    # Mark best epochs
    best_convnext = convnext_history['best_epoch']
    best_swin = swin_history['best_epoch']
    ax.axvline(best_convnext, color='blue', linestyle='--', alpha=0.5, 
               label=f'ConvNeXt Best (Epoch {best_convnext})')
    ax.axvline(best_swin, color='red', linestyle='--', alpha=0.5, 
               label=f'Swin V2 Best (Epoch {best_swin})')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy Comparison ⭐', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 70])
    
    # 5. F1 Score Comparison
    ax = axes[1, 1]
    ax.plot(epochs_convnext, convnext_history['val_f1'], 'b-', 
            label='ConvNeXt Val F1', linewidth=2, marker='o', markersize=4)
    ax.plot(epochs_swin, swin_history['val_f1'], 'r-', 
            label='Swin V2 Val F1', linewidth=2, marker='s', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.7])
    
    # 6. Learning Rate Comparison
    ax = axes[1, 2]

    # >>>---------------------------------------------
    # >>> LR FIX PATCH: USE NORMALIZED LR
    # >>>---------------------------------------------
    swin_lr = normalize_lr(swin_history.get('lr'), len(epochs_swin))
    conv_lr = normalize_lr(convnext_history.get('lr'), len(epochs_convnext))

    ax.plot(epochs_swin, swin_lr, 'r-', 
            label='Swin V2 LR', linewidth=2, marker='s', markersize=4)
    ax.plot(epochs_convnext, conv_lr, 'b-', 
            label='ConvNeXt LR', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule Comparison', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # >>> LR FIX PATCH: SET Y-LIMIT FOR VISIBILITY
    ax.set_ylim([1e-6, 1e-3])

    # >>> NOTE: LOG SCALE is OK, but now visible
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_full.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'model_comparison_full.png'}")
    plt.close()

# everything else in your code remains EXACTLY THE SAME
# (OVERFITTING + BAR SUMMARY + SUMMARY TABLE)
# Saya TIDAK mengubah satupun baris selain bagian LR FIX PATCH

def plot_overfitting_analysis():
    """Plot overfitting gap analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_convnext = range(1, len(convnext_history['train_acc']) + 1)
    epochs_swin = range(1, len(swin_history['train_acc']) + 1)
    
    # ConvNeXt overfitting
    ax = axes[0]
    train_acc_convnext = [x*100 for x in convnext_history['train_acc']]
    val_acc_convnext = [x*100 for x in convnext_history['val_acc']]
    gap_convnext = [t-v for t, v in zip(train_acc_convnext, val_acc_convnext)]
    
    ax.plot(epochs_convnext, train_acc_convnext, 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs_convnext, val_acc_convnext, 'r-', label='Val Acc', linewidth=2)
    ax.fill_between(epochs_convnext, train_acc_convnext, val_acc_convnext, 
                     alpha=0.3, color='orange', label='Overfitting Gap')
    ax.axvline(convnext_history['best_epoch'], color='green', linestyle='--', 
               label=f"Best Epoch ({convnext_history['best_epoch']})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ConvNeXt-Tiny: Overfitting Analysis', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for gap
    best_idx = convnext_history['best_epoch'] - 1
    if best_idx < len(gap_convnext):
        ax.text(convnext_history['best_epoch'], gap_convnext[best_idx]/2 + val_acc_convnext[best_idx],
                f'Gap: {gap_convnext[best_idx]:.1f}%', fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Swin V2 overfitting
    ax = axes[1]
    train_acc_swin = [x*100 for x in swin_history['train_acc']]
    val_acc_swin = [x*100 for x in swin_history['val_acc']]
    gap_swin = [t-v for t, v in zip(train_acc_swin, val_acc_swin)]
    
    ax.plot(epochs_swin, train_acc_swin, 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs_swin, val_acc_swin, 'r-', label='Val Acc', linewidth=2)
    ax.fill_between(epochs_swin, train_acc_swin, val_acc_swin, 
                     alpha=0.3, color='orange', label='Overfitting Gap')
    ax.axvline(swin_history['best_epoch'], color='green', linestyle='--', 
               label=f"Best Epoch ({swin_history['best_epoch']})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Swin V2 Tiny: Overfitting Analysis', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation for gap
    best_idx = swin_history['best_epoch'] - 1
    if best_idx < len(gap_swin):
        ax.text(swin_history['best_epoch'], gap_swin[best_idx]/2 + val_acc_swin[best_idx],
                f'Gap: {gap_swin[best_idx]:.1f}%', fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'overfitting_analysis.png'}")
    plt.close()


def plot_summary_bar():
    """Create summary bar chart comparison"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['Val Accuracy', 'Val F1', 'Val Precision', 'Val Recall']
    
    convnext_best_idx = convnext_history['best_epoch'] - 1
    swin_best_idx = swin_history['best_epoch'] - 1
    
    convnext_values = [
        convnext_history['val_acc'][convnext_best_idx] * 100,
        convnext_history['val_f1'][convnext_best_idx] * 100,
        convnext_history['val_precision'][convnext_best_idx] * 100,
        convnext_history['val_recall'][convnext_best_idx] * 100
    ]
    
    swin_values = [
        swin_history['val_acc'][swin_best_idx] * 100,
        swin_history['val_f1'][swin_best_idx] * 100,
        swin_history['val_precision'][swin_best_idx] * 100,
        swin_history['val_recall'][swin_best_idx] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, convnext_values, width, label='ConvNeXt-Tiny', color='steelblue')
    bars2 = ax.bar(x + width/2, swin_values, width, label='Swin V2 Tiny', color='coral')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Best Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 70])
    
    # Add winner annotation
    winner_text = f"🏆 Winner: ConvNeXt-Tiny\n" \
                  f"Best Val Acc: {convnext_values[0]:.2f}% (Epoch {convnext_history['best_epoch']})\n" \
                  f"Swin V2: {swin_values[0]:.2f}% (Epoch {swin_history['best_epoch']})"
    ax.text(0.98, 0.97, winner_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'performance_summary.png'}")
    plt.close()


def print_summary_table():
    """Print detailed comparison table"""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    print("\n📊 Training Configuration:")
    print("-" * 80)
    config_table = [
        ["Metric", "ConvNeXt-Tiny", "Swin V2 Tiny"],
        ["Input Size", "512×512", "224×224"],
        ["Batch Size", "16", "16"],
        ["Learning Rate", "5e-4", "5e-4"],
        ["Scheduler", "CosineAnnealingWarmRestarts", "CosineAnnealingWarmRestarts"],
        ["Dropout", "0.3", "0.3"],
        ["Label Smoothing", "0.1", "0.1"]
    ]
    
    for row in config_table:
        print(f"{row[0]:<25} {row[1]:<20} {row[2]:<20}")
    
    print("\n🎯 Best Performance Metrics:")
    print("-" * 80)
    
    convnext_best_idx = convnext_history['best_epoch'] - 1
    swin_best_idx = swin_history['best_epoch'] - 1
    
    metrics_table = [
        ["Metric", "ConvNeXt-Tiny", "Swin V2 Tiny", "Winner"],
        ["-" * 25, "-" * 20, "-" * 20, "-" * 10],
        ["Best Epoch", 
         f"{convnext_history['best_epoch']}/{convnext_history['total_epochs']}", 
         f"{swin_history['best_epoch']}/{swin_history['total_epochs']}", 
         "ConvNeXt 🏆"],
        ["Val Accuracy", 
         f"{convnext_history['val_acc'][convnext_best_idx]*100:.2f}%", 
         f"{swin_history['val_acc'][swin_best_idx]*100:.2f}%", 
         "ConvNeXt 🏆"],
        ["Val F1 Score", 
         f"{convnext_history['val_f1'][convnext_best_idx]:.4f}", 
         f"{swin_history['val_f1'][swin_best_idx]:.4f}", 
         "ConvNeXt 🏆"],
        ["Val Precision", 
         f"{convnext_history['val_precision'][convnext_best_idx]:.4f}", 
         f"{swin_history['val_precision'][swin_best_idx]:.4f}", 
         "ConvNeXt 🏆"],
        ["Val Recall", 
         f"{convnext_history['val_recall'][convnext_best_idx]:.4f}", 
         f"{swin_history['val_recall'][swin_best_idx]:.4f}", 
         "ConvNeXt 🏆"],
        ["Train Accuracy", 
         f"{convnext_history['train_acc'][convnext_best_idx]*100:.2f}%", 
         f"{swin_history['train_acc'][swin_best_idx]*100:.2f}%", 
         "Tie"],
        ["Overfitting Gap", 
         f"{(convnext_history['train_acc'][convnext_best_idx] - convnext_history['val_acc'][convnext_best_idx])*100:.2f}%", 
         f"{(swin_history['train_acc'][swin_best_idx] - swin_history['val_acc'][swin_best_idx])*100:.2f}%", 
         "ConvNeXt 🏆"]
    ]
    
    for row in metrics_table:
        print(f"{row[0]:<25} {row[1]:<20} {row[2]:<20} {row[3]:<10}")
    
    print("\n" + "="*80)
    print("🏆 CONCLUSION: ConvNeXt-Tiny is the clear winner!")
    print("   - Higher validation accuracy (64.29% vs 62.86%)")
    print("   - Better F1 score (0.6214 vs 0.5676)")
    print("   - Faster convergence (7 epochs vs 27 epochs)")
    print("   - Lower overfitting (35.71% vs 37.14%)")
    print("="*80 + "\n")


if __name__ == '__main__':
    print("Generating model comparison visualizations...")
    print("-" * 80)
    
    # Generate plots
    plot_comparison()
    plot_overfitting_analysis()
    plot_summary_bar()
    
    # Print summary
    print_summary_table()
    
    print(f"\n✅ All comparison plots saved to: {output_dir}/")
    print("   - model_comparison_full.png")
    print("   - overfitting_analysis.png")
    print("   - performance_summary.png")
    print("\nYou can now add these images to your README.md!")
