import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny model untuk face recognition
    Menggunakan ConvNeXt-Tiny pretrained weights dari ImageNet-1K V1
    Input size: 512x512
    """
    def __init__(self, num_classes=70, pretrained=True, dropout=0.2):
        super(ConvNeXtClassifier, self).__init__()
        
        # Load pretrained ConvNeXt-Tiny model
        # ConvNeXt-Tiny memiliki 28M parameters
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            backbone = convnext_tiny(weights=weights)
        else:
            backbone = convnext_tiny(weights=None)
        
        # Remove the final classification layer
        # ConvNeXt structure: features + avgpool + classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # ConvNeXt-Tiny output embedding size is 768, reduce to 128
        self.embedding_size = 768
        
        # Embedding reduction layer
        self.embed_reduce = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(768, 128),
            nn.ReLU()
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(128, eps=1e-6),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        # Extract features from ConvNeXt backbone
        # Input: (B, 3, 512, 512) -> Features: (B, 768, H, W)
        x = self.features(x)
        
        # Global average pooling: (B, 768, H, W) -> (B, 768, 1, 1)
        x = self.avgpool(x)
        
        # Reduce embedding dimension: (B, 768, 1, 1) -> (B, 128)
        x = self.embed_reduce(x)
        
        # Classification: (B, 128) -> (B, num_classes)
        output = self.head(x)
        
        return output
    
    def freeze_backbone(self):
        """Freeze backbone features, hanya train classification head"""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.avgpool.parameters():
            param.requires_grad = False
        print("✓ Backbone frozen, only training classification head")
    
    def unfreeze_backbone(self):
        """Unfreeze semua layers untuk fine-tuning"""
        for param in self.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen, training all layers")
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes=70, pretrained=True, dropout=0.2, device='cuda'):
    """
    Create ConvNeXt-Tiny model
    
    Args:
        num_classes (int): Jumlah kelas untuk klasifikasi
        pretrained (bool): Menggunakan pretrained weights atau tidak
        dropout (float): Dropout rate untuk classification head
        device (str): Device untuk model ('cuda' atau 'cpu')
    
    Returns:
        model: ConvNeXt model yang sudah di-load ke device
    """
    model = ConvNeXtClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    return model.to(device)


if __name__ == "__main__":
    print("="*60)
    print("Testing ConvNeXt-Tiny Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    model = create_model(num_classes=70, pretrained=True, device=device)
    
    print("\nModel Information:")
    print(f"  Architecture  : ConvNeXt-Tiny")
    print(f"  Pretrained    : ImageNet-1K V1")
    print(f"  Embedding size: 768")
    print(f"  Total params  : {model.get_num_total_params():,} ({model.get_num_total_params()/1e6:.2f}M)")
    print(f"  Trainable     : {model.get_num_trainable_params():,} ({model.get_num_trainable_params()/1e6:.2f}M)")
    
    print("\nTesting forward pass with 512x512 input...")
    # Test with 512x512 input
    dummy_input = torch.randn(4, 3, 512, 512).to(device)
    
    # Test training mode
    model.train()
    with torch.no_grad():
        output_train = model(dummy_input)
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_input)
    
    print(f"  Input shape       : {dummy_input.shape}")
    print(f"  Output shape      : {output_train.shape}")
    print(f"  Output shape (eval): {output_eval.shape}")
    
    print("\n✓ ConvNeXt-Tiny model ready for 512x512 input!")
    print("="*60)
