import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import nibabel as nib

from DenseNetEnsemble import DenseNet3D

# Load the saved DenseNet3D model without the final classification layer
# Function to load the saved DenseNet model
def load_densenet_model(model_path, device):
    model = DenseNet3D()  # Initialize your DenseNet3D model
    state_dict = torch.load(model_path, map_location=device)  # Load the .pth file
    
    # Remove `module.` from the keys if the model was saved with DataParallel
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove 'module.' prefix
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)  # Load the modified state_dict
    model.to(device)  # Move model to device (GPU or CPU)
    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {model_path}")
    return model

# Function to load and preprocess the MRI images
def load_and_preprocess_mri(image_path):
    mri_image = nib.load(image_path)
    mri_image_data = mri_image.get_fdata()
    # Normalize the MRI data
    mri_image_data = (mri_image_data - np.min(mri_image_data)) / (np.max(mri_image_data) - np.min(mri_image_data))
    mri_image_tensor = torch.tensor(mri_image_data, dtype=torch.float32)
    return mri_image_tensor

# Custom dataset class to handle MRI images and their IDs
class MRIDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        # Automatically collect all .nii.gz files in the image directory
        self.mri_ids = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.nii.gz')]

    def __len__(self):
        return len(self.mri_ids)

    def __getitem__(self, idx):
        mri_id = self.mri_ids[idx]
        image_path = os.path.join(self.image_dir, f"{mri_id}.nii.gz")
        image_tensor = load_and_preprocess_mri(image_path)
        return image_tensor, mri_id

# Function to extract features from the DenseNet model
def extract_features_from_densenet(densenet_model, dataloader, device):
    densenet_model.eval()  # Ensure the model is in evaluation mode
    features_list = []
    mri_id_list = []

    with torch.no_grad():  # Disable gradient calculations for inference
        for images, mri_ids in dataloader:
            images = images.unsqueeze(1).to(device)  # Add channel dimension and move to the appropriate device
            # Get the DenseNet features
            features = densenet_model(images)
            features_np = features.cpu().numpy()  # Move features to CPU and convert to NumPy array
            features_list.append(features_np)
            mri_id_list.extend(mri_ids)

    # Stack features into one array
    features_array = np.vstack(features_list)
    return features_array, mri_id_list

# Function to save features and MRI IDs to CSV
def save_features_to_csv(features, mri_ids, output_csv_path):
    # Create a DataFrame with the MRI_IDs and their corresponding features
    df_features = pd.DataFrame(features)
    df_features['MRI_ID'] = mri_ids

    # Reorder the columns to have MRI_ID as the first column
    cols = df_features.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df_features = df_features[cols]

    # Save the DataFrame to CSV
    df_features.to_csv(output_csv_path, index=False)
    print(f"Features saved to {output_csv_path}")

# Main script to load model, extract features, and save to CSV
if __name__ == "__main__":
    model_path = "/db/proj-mag/dense_model_features.pth"  # Path to your trained DenseNet model without final classification layer
    image_dir = "/db/proj-mag/final"  # Directory containing the MRI images (.nii.gz files)
    output_csv_path = "/db/proj-mag/output_features.csv"  # Path to save extracted features CSV

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU

    # Load the pre-trained DenseNet model without final classification layer
    densenet_model = load_densenet_model(model_path, device)

    # Create a dataset and dataloader
    dataset = MRIDataset(image_dir=image_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)  # Adjust batch_size and num_workers as needed

    # Extract features from DenseNet
    features, mri_ids = extract_features_from_densenet(densenet_model, dataloader, device)

    # Save the extracted features to CSV
    save_features_to_csv(features, mri_ids, output_csv_path)