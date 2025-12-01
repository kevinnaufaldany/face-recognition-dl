import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


class FaceDataset(Dataset):
    """
    Dataset class untuk membaca gambar dari folder Train
    Setiap subfolder merepresentasikan satu class
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path ke folder Train
            transform (callable, optional): Optional transform untuk augmentasi
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Baca semua path gambar dan labelnya
        self.samples = []
        skipped_files = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    # Validasi dan perbaiki file corrupt
                    try:
                        # Try to open and verify
                        with Image.open(img_path) as img:
                            img.verify()
                        self.samples.append((str(img_path), self.class_to_idx[class_name]))
                    except Exception:
                        # Repair corrupt file
                        try:
                            # Load without verify, convert, and save
                            img = Image.open(img_path)
                            img = img.convert('RGB')
                            # Save as jpg if webp
                            if img_path.suffix.lower() == '.webp':
                                new_path = img_path.with_suffix('.jpg')
                                img.save(new_path, quality=95)
                                self.samples.append((str(new_path), self.class_to_idx[class_name]))
                            else:
                                img.save(img_path, quality=95)
                                self.samples.append((str(img_path), self.class_to_idx[class_name]))
                        except Exception:
                            skipped_files.append(img_path.name)
        
        if skipped_files:
            print(f"⚠ {len(skipped_files)} file corrupt di-skip")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_num_classes(self):
        return len(self.classes)


def get_transforms(img_size=224, is_training=True):
    """
    Mendefinisikan augmentasi untuk training dan validation/test
    
    Args:
        img_size (int): Ukuran gambar (224 untuk Swin/DeiT, 512 untuk ConvNeXt/ArcFace)
        is_training (bool): True untuk training transforms, False untuk val/test
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random')
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(data_dir, img_size=224, batch_size=8, num_workers=0, seed=42):
    """
    Membuat DataLoader untuk train dan validation
    Split: 1 foto per kelas untuk validation, sisanya (3 foto) untuk training
    
    Args:
        data_dir (string): Path ke folder Train
        img_size (int): Ukuran gambar (default 224 untuk Swin/DeiT, 512 untuk ConvNeXt)
        batch_size (int): Batch size untuk DataLoader
        num_workers (int): Number of workers untuk DataLoader
        seed (int): Random seed untuk reproducibility
    
    Returns:
        tuple: (train_loader, val_loader, num_classes, class_names)
    """
    # Set random seed untuk reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Buat dataset tanpa transform dulu
    temp_dataset = FaceDataset(data_dir, transform=None)
    num_classes = temp_dataset.get_num_classes()
    
    # Kelompokkan samples berdasarkan class
    class_samples = {}
    for idx, (img_path, label) in enumerate(temp_dataset.samples):
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(idx)
    
    # Split: 1 foto per kelas untuk val, sisanya untuk train
    train_indices = []
    val_indices = []
    
    train_per_class = []
    val_per_class = []
    
    for label in sorted(class_samples.keys()):
        indices = class_samples[label]
        # Shuffle indices untuk class ini
        np.random.seed(seed + label)  # Seed berbeda per class
        shuffled = np.random.permutation(indices).tolist()
        
        # Pastikan setiap kelas punya minimal 4 foto
        num_samples = len(shuffled)
        if num_samples >= 4:
            # 1 foto untuk val, 3 untuk train
            val_indices.append(shuffled[0])
            train_indices.extend(shuffled[1:4])
            train_per_class.append(3)
            val_per_class.append(1)
        elif num_samples > 1:
            # Jika kurang dari 4, ambil 1 untuk val, sisanya train
            val_indices.append(shuffled[0])
            train_indices.extend(shuffled[1:])
            train_per_class.append(len(shuffled) - 1)
            val_per_class.append(1)
        else:
            # Jika cuma 1, masukkan ke train
            train_indices.extend(shuffled)
            train_per_class.append(1)
            val_per_class.append(0)
    
    total_images = len(temp_dataset.samples)
    print(f"✓ Dataset loaded: {num_classes} classes, {total_images} total images")
    print(f"  Split: {len(train_indices)} train ({len(train_indices)/total_images*100:.1f}%), "
          f"{len(val_indices)} val ({len(val_indices)/total_images*100:.1f}%)")
    
    # Buat dataset dengan transform yang sesuai
    train_dataset = FaceDataset(data_dir, transform=get_transforms(img_size=img_size, is_training=True))
    val_dataset = FaceDataset(data_dir, transform=get_transforms(img_size=img_size, is_training=False))
    
    # Subset berdasarkan indices
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Buat DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names for confusion matrix
    class_names = temp_dataset.classes
    
    return train_loader, val_loader, num_classes, class_names


if __name__ == "__main__":
    # Test datareader
    data_dir = "dataset/Train"
    
    print("="*60)
    print("Testing FaceDataset DataLoader")
    print("="*60)
    
    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir, 
        batch_size=16,
        num_workers=0  # Set to 0 untuk testing
    )
    
    print(f"\nJumlah classes: {num_classes}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test satu batch
    print("\nTesting one batch dari train_loader:")
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"Label range: {labels.min().item()} - {labels.max().item()}")
    print("\n" + "="*60)
