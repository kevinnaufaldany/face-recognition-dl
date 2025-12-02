import os
import csv

# Define paths
test_folder = "Test"
output_file = "jawaban.csv"

# Collect all image paths
image_paths = []

for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.lower().endswith(('.jpeg', '.jpg', '.png', '.gif', 'bmp', '.webp')):
            full_path = os.path.join(root, file)
            # Convert to relative path from Test folder
            relative_path = os.path.relpath(full_path, test_folder)
            image_paths.append(relative_path)

# Write to CSV with filename and empty label
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])  # Header
    
    for path in image_paths:
        writer.writerow([path, ''])  # filename and empty label

print(f"Total images: {len(image_paths)}")
print(f"CSV file created: {output_file}")