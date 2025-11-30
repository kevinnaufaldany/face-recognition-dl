import torch
import torch.nn as nn
import timm


class DeiTClassifier(nn.Module):
    """
    DeiT-Small model untuk face recognition
    Menggunakan DeiT-Small pretrained weights dari Facebook/ImageNet-1K
    Input size: 512x512
    """
    def __init__(self, num_classes=70, pretrained=True, dropout=0.2):
        super(DeiTClassifier, self).__init__()
        
        # Load pretrained DeiT-Small model
        # deit_small_distilled_patch16_224 memiliki 22M parameters
        if pretrained:
            self.backbone = timm.create_model(
                'deit_small_distilled_patch16_224.fb_in1k',
                pretrained=True,
                num_classes=0,  # Remove classification head
                img_size=512    # Change input size to 512x512
            )
        else:
            self.backbone = timm.create_model(
                'deit_small_distilled_patch16_224.fb_in1k',
                pretrained=False,
                num_classes=0,
                img_size=512
            )
        
        # DeiT-Small output embedding size is 384, reduce to 128
        self.embedding_size = 384
        
        # Embedding reduction layer
        self.embed_reduce = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU()
        )
        
        # Classification head
        self.head = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, x):
        # Extract features from DeiT backbone
        # DeiT akan otomatis handle input 512x512 (adaptive pooling internal)
        features = self.backbone(x)  # (B, 384)
        
        # Reduce embedding dimension: 384 -> 128
        features = self.embed_reduce(features)  # (B, 128)
        
        # Classification
        output = self.head(features)
        
        return output
    
    def freeze_backbone(self):
        """Freeze backbone, hanya train classification head"""
        for param in self.backbone.parameters():
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
    Create DeiT-Small model
    
    Args:
        num_classes (int): Jumlah kelas untuk klasifikasi
        pretrained (bool): Menggunakan pretrained weights atau tidak
        dropout (float): Dropout rate untuk classification head
        device (str): Device untuk model ('cuda' atau 'cpu')
    
    Returns:
        model: DeiT model yang sudah di-load ke device
    """
    model = DeiTClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    return model.to(device)


if __name__ == "__main__":
    print("="*60)
    print("Testing DeiT-Small Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create model
    model = create_model(num_classes=70, pretrained=True, device=device)
    
    print("\nModel Information:")
    print(f"  Architecture  : DeiT-Small (Distilled)")
    print(f"  Pretrained    : Facebook/ImageNet-1K")
    print(f"  Embedding size: 384")
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
    
    print("\n✓ DeiT-Small model ready for 512x512 input!")
    print("="*60)
