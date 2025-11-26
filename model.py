import torch
import torch.nn as nn
from torchvision.models import swin_v2_s, Swin_V2_S_Weights


class SwinV2Classifier(nn.Module):
    """
    Swin Transformer V2 Small untuk image classification
    Menggunakan pretrained weights dari ImageNet-1K
    """
    def __init__(self, num_classes=70, pretrained=True):
        """
        Args:
            num_classes (int): Jumlah kelas untuk klasifikasi
            pretrained (bool): Menggunakan pretrained weights atau tidak
        """
        super(SwinV2Classifier, self).__init__()
        
        # Load pretrained Swin-V2-S model
        if pretrained:
            weights = Swin_V2_S_Weights.IMAGENET1K_V1
            self.model = swin_v2_s(weights=weights)
        else:
            self.model = swin_v2_s(weights=None)
        
        # Ganti head classifier sesuai jumlah kelas
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor dengan shape (batch_size, 3, 224, 224)
        
        Returns:
            torch.Tensor: Output logits dengan shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def freeze_backbone(self):
        """
        Freeze semua layer kecuali classifier head
        Berguna untuk fine-tuning dengan frozen features
        """
        for name, param in self.model.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("✓ Backbone frozen, only training classifier head")
    
    def unfreeze_backbone(self):
        """
        Unfreeze semua layer untuk full fine-tuning
        """
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen, training all layers")
    
    def get_num_trainable_params(self):
        """
        Menghitung jumlah parameter yang trainable
        
        Returns:
            int: Jumlah trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self):
        """
        Menghitung total jumlah parameter
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters())


def create_model(num_classes=70, pretrained=True, device='cuda'):
    """
    Helper function untuk membuat model dan memindahkannya ke device
    
    Args:
        num_classes (int): Jumlah kelas
        pretrained (bool): Menggunakan pretrained weights
        device (str): Device untuk model ('cuda' atau 'cpu')
    
    Returns:
        SwinV2Classifier: Model yang sudah siap digunakan
    """
    model = SwinV2Classifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test model
    print("="*60)
    print("Testing Swin-V2-S Model")
    print("="*60)
    
    # Cek CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice available: {device}")
    
    # Buat model
    model = create_model(num_classes=70, pretrained=True, device=device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test freeze/unfreeze
    print("\nTesting freeze/unfreeze...")
    model.freeze_backbone()
    print(f"Trainable params after freeze: {model.get_num_trainable_params():,}")
    
    model.unfreeze_backbone()
    print(f"Trainable params after unfreeze: {model.get_num_trainable_params():,}")
    
    print("\n" + "="*60)
