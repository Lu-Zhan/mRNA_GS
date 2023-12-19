from PIL import Image
import os

# Directory path
dir_path = '/media/mldadmin/home/s122mdg39_05/Projects_mrna/data/1213_demo_data_v2/raw1'
tgt_path = f'/media/mldadmin/home/s122mdg39_05/Projects_mrna/data/1213_demo_data_v2/{os.path.basename(dir_path)}_cropped'

# Cropping rectangle (left, upper, right, lower)
crop_rectangle = (140, 280, 260, 400)

# Iterate through each file in the directory
for filename in os.listdir(dir_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        full_path = os.path.join(dir_path, filename)

        # Open the image
        with Image.open(full_path) as img:
            # Crop the image
            cropped_img = img.crop(crop_rectangle)

            # Save the cropped image
            save_path = os.path.join(tgt_path, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cropped_img.save(save_path)

print("Cropping completed.")
