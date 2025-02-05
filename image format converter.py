import os
import nibabel as nib
import numpy as np
from PIL import Image

# Define source and destination folders
source_folder = r'C:\Users\cl501_29\Desktop\Vamsidhar\Liver Segmentation GA_ MV Project\Task03_Liver\labelsTr'
destination_parent_folder = r'C:\Users\cl501_29\Desktop\Vamsidhar\Liver Segmentation GA_ MV Project\Task03_Liver\labelsTr - JPG'

# Iterate over all folders in the source folder
for subfolder in os.listdir(source_folder):
    subfolder_path = os.path.join(source_folder, subfolder)

    # Check if it's a folder
    if os.path.isdir(subfolder_path):
        # Iterate over all .nii files in the subfolder
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith('.nii') and not file_name.startswith("._"):
                # Load the .nii file
                nii_file_path = os.path.join(subfolder_path, file_name)

                # Check if the file is non-empty
                if os.path.getsize(nii_file_path) > 0:
                    try:
                        nii_data = nib.load(nii_file_path).get_fdata()

                        # Normalize the data to 0-255 range for visualization
                        nii_data = ((nii_data - np.min(nii_data)) / (np.max(nii_data) - np.min(nii_data)) * 255).astype(
                            np.uint8)

                        # Define the destination folder (same name as the source subfolder)
                        destination_folder = os.path.join(destination_parent_folder, subfolder)
                        os.makedirs(destination_folder, exist_ok=True)

                        # Save each slice as a separate .jpg file
                        for i in range(nii_data.shape[2]):  # Iterate over slices
                            slice_image = Image.fromarray(nii_data[:, :, i])
                            slice_file_name = f"{os.path.splitext(file_name)[0]}_slice_{i + 1}.jpg"
                            slice_file_path = os.path.join(destination_folder, slice_file_name)
                            slice_image.save(slice_file_path)

                        print(f"Converted {file_name} to JPG slices in {destination_folder}")
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")
                else:
                    print(f"Skipped empty file: {file_name}")

print("Conversion complete!")
