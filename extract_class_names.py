"""
Extract class names from dataset folder structure
This script reads folder names from dataset/Train and saves them to class_names.txt
"""

import os
from pathlib import Path

def extract_class_names(dataset_path='dataset/Train', output_file='class_names.txt'):
    """
    Extract class names from dataset folders
    
    Args:
        dataset_path (str): Path to dataset directory containing class folders
        output_file (str): Output file to save class names
    
    Returns:
        list: List of class names
    """
    
    # Get absolute path
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory '{dataset_path}' does not exist")
        return None
    
    # Get all subdirectories (class folders)
    class_folders = sorted([
        folder.name for folder in dataset_dir.iterdir() 
        if folder.is_dir() and not folder.name.startswith('.')
    ])
    
    # Filter out __MACOSX and other system folders
    class_names = [
        name for name in class_folders 
        if not name.startswith('__') and name != '.DS_Store'
    ]
    
    num_classes = len(class_names)
    
    print(f"📊 Found {num_classes} classes")
    print(f"💾 Saving to {output_file}...\n")
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, class_name in enumerate(class_names, 1):
            f.write(f"{class_name}\n")
            print(f"  {idx:2d}. {class_name}")
    
    print(f"\n✅ Successfully saved {num_classes} class names to {output_file}")
    return class_names


def display_class_names(output_file='class_names.txt'):
    """
    Display saved class names from file
    
    Args:
        output_file (str): File to read class names from
    """
    if not os.path.exists(output_file):
        print(f"❌ File '{output_file}' not found")
        return
    
    with open(output_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    print(f"\n📋 Class names from {output_file}:")
    print(f"Total: {len(class_names)} classes\n")
    
    for idx, class_name in enumerate(class_names, 1):
        print(f"  {idx:2d}. {class_name}")


if __name__ == '__main__':
    import sys
    
    # Path to dataset
    dataset_path = 'dataset/Train'
    output_file = 'class_names.txt'
    
    print("="*60)
    print("Extract Class Names from Dataset")
    print("="*60 + "\n")
    
    # Extract and save class names
    class_names = extract_class_names(dataset_path, output_file)
    
    if class_names:
        # Display summary
        print("\n" + "="*60)
        print("Summary:")
        print("="*60)
        print(f"✓ Dataset path: {dataset_path}")
        print(f"✓ Output file: {output_file}")
        print(f"✓ Total classes: {len(class_names)}")
        print("="*60)
    else:
        print("\n❌ Failed to extract class names")
        sys.exit(1)
