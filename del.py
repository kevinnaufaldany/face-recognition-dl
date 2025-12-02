import os
from pathlib import Path

# Tentukan path folder test
test_folder = "Test/"

# Ekstensi file gambar yang akan dihapus
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Iterasi setiap subfolder
for root, dirs, files in os.walk(test_folder):
    # Ambil semua file gambar di subfolder ini
    image_files = [f for f in files 
                   if Path(f).suffix.lower() in image_extensions]
    
    # Jika ada file gambar, hapus semua
    if len(image_files) > 0:
        for file in image_files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Terhapus: {file_path}")
        
        print(f"Subfolder {root} - Total dihapus: {len(image_files)}\n")
