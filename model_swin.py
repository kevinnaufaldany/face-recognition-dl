"""
Swin Transformer V2 Tiny Model for Face Recognition
Input: 224x224 (preprocessed dengan MediaPipe face cropping)
Pretrained: IMAGENET1K_V1
Architecture matches ConvNeXt structure for consistency
"""

import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


class SwinV2Classifier(nn.Module):
    """
    Swin Transformer V2 Tiny untuk face recognition
    Menggunakan Swin V2 Tiny pretrained weights dari ImageNet-1K V1
    Input size: 224x224
    """
    
    def __init__(self, num_classes=70, pretrained=True, dropout=0.3):
        super(SwinV2Classifier, self).__init__()
        
        # Load pretrained Swin V2 Tiny
        # Swin V2 Tiny memiliki ~28M parameters
        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1
            backbone = swin_v2_t(weights=weights)
            print("✓ Loaded Swin V2 Tiny with IMAGENET1K_V1 pretrained weights")
        else:
            backbone = swin_v2_t(weights=None)
            print("✗ Training from scratch (no pretrained weights)")
        
        # Get feature dimension from backbone
        # Swin V2 Tiny output: 768 dimensions (use full dimensions)
        self.embedding_size = 768
        
        # Remove original classifier head
        self.features = backbone  # Keep entire backbone
        self.features.head = nn.Identity()  # Remove original head
        
        # Classification head (directly use 768 dimensions)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )
        
        self.num_classes = num_classes
        
        # Initialize new layers
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classification head"""
        for layer in self.head.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight, 1.0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            output: Classification output [B, num_classes]
        """
        # Extract features dengan backbone
        # Input: (B, 3, 224, 224) -> Features: (B, 768)
        x = self.features(x)
        
        # Classification: (B, 768) -> (B, num_classes)
        output = self.head(x)
        
        return output
    
    def freeze_backbone(self):
        """Freeze backbone features, hanya train classification head"""
        for param in self.features.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen, only training classification head")
    
    def unfreeze_backbone(self):
        """Unfreeze semua layers untuk fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen, training all layers")
    
    def get_embedding(self, x):
        """
        Get embedding vector (untuk feature extraction)
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            embedding: Feature vector [B, 128]
        """
        with torch.no_grad():
            features = self.features(x)
            embedding = self.embed_reduce(features)
        return embedding
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes=70, pretrained=True, dropout=0.3, device='cuda'):
    """
    Create Swin V2 Tiny model
    
    Args:
        num_classes (int): Jumlah kelas untuk klasifikasi
        pretrained (bool): Menggunakan pretrained weights atau tidak
        dropout (float): Dropout rate untuk classification head
        device (str): Device untuk model ('cuda' atau 'cpu')
        
    Returns:
        model: Swin V2 model yang sudah di-load ke device
    """
    model = SwinV2Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    return model.to(device)


def count_parameters(model):
    """Count total dan trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    """Test model"""
    print("="*60)
    print("Testing Swin V2 Tiny Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    model = create_model(num_classes=70, pretrained=True, dropout=0.3, device=device)
    
    print("\nModel Information:")
    print(f"  Architecture  : Swin Transformer V2 Tiny")
    print(f"  Pretrained    : IMAGENET1K_V1")
    print(f"  Input size    : 224x224")
    print(f"  Embedding size: 128 (reduced from 768)")
    print(f"  Total params  : {model.get_num_total_params():,} ({model.get_num_total_params()/1e6:.2f}M)")
    print(f"  Trainable     : {model.get_num_trainable_params():,} ({model.get_num_trainable_params()/1e6:.2f}M)")
    
    print("\nTesting forward pass with 224x224 input...")
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Test training mode
    model.train()
    with torch.no_grad():
        output_train = model(x)
        print(f"  Train mode - Input: {x.shape} -> Output: {output_train.shape}")
    
    # Test eval mode
    model.eval()
    with torch.no_grad():
        output_eval = model(x)
        embedding = model.get_embedding(x)
        print(f"  Eval mode  - Input: {x.shape} -> Output: {output_eval.shape}")
        print(f"  Embedding  - Shape: {embedding.shape}")
    
    # Test freeze/unfreeze
    print("\nTesting freeze/unfreeze:")
    model.freeze_backbone()
    print(f"  Trainable after freeze: {model.get_num_trainable_params():,}")
    
    model.unfreeze_backbone()
    print(f"  Trainable after unfreeze: {model.get_num_trainable_params():,}")
    
    print("\n✓ Model test passed!")
    print("="*60)

