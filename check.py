import os
from pathlib import Path
from PIL import Image

def count_valid_images(folder):
    """Menghitung jumlah gambar valid dalam sebuah folder."""
    valid_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    count = 0
    corrupted = []

    for file in folder.iterdir():
        if file.suffix.lower() in valid_ext:
            try:
                img = Image.open(file)
                img.load()   # lebih aman daripada verify()
                count += 1
            except Exception as e:
                corrupted.append(file.name)

    return count, corrupted


def inspect_dataset(root_dir="dataset/Train"):
    root = Path(root_dir)

    if not root.exists():
        print(f"Folder '{root_dir}' tidak ditemukan!")
        return

    print("=" * 60)
    print(f"📁 Inspecting dataset in: {root_dir}")
    print("=" * 60)

    class_folders = sorted([d for d in root.iterdir() if d.is_dir()])
    summary = []

    for class_dir in class_folders:
        total_imgs, corrupted = count_valid_images(class_dir)

        # Tentukan split (1 validation, sisanya train)
        if total_imgs >= 4:
            val = 1
            train = total_imgs - 1
        elif total_imgs > 1:
            val = 1
            train = total_imgs - 1
        else:
            val = 0
            train = total_imgs

        summary.append((class_dir.name, total_imgs, train, val, corrupted))

    # Print hasil
    print(f"{'Class Name':30s} | Total | Train | Val")
    print("-" * 70)

    for class_name, total, train, val, corrupted in summary:
        print(f"{class_name:30s} | {total:5d} | {train:5d} | {val:3d}")

        if corrupted:
            print(f"   ⚠ Corrupted / unreadable files: {corrupted}")

    print("\nSelesai cek dataset.")
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset("dataset/Train")
