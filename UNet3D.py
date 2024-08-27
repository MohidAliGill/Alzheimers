import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchio as tio
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

augmentation_pipeline = tio.Compose([
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5),
    tio.RandomFlip(axes=(0, 1, 2)),
    tio.OneOf({
        tio.RandomMotion(p=0.2): 0.1,  # Adjust probabilities based on class
        tio.RandomNoise(p=0.2): 0.1,
        tio.RandomBiasField(p=0.2): 0.1  # Add more types of augmentations
    }),
    tio.RescaleIntensity((-1, 1))  # Ensure normalization is consistent
])

# Utility function to load and check MRI dimensions and channels
def load_mri_image_and_check_channels(image_path):
    mri_image = nib.load(image_path)
    mri_image_data = mri_image.get_fdata()
    dimensions = mri_image_data.shape
    return dimensions

# Function to load and normalize MRI images
def load_and_preprocess_mri(image_path):
    mri_image = nib.load(image_path)
    mri_image_data = mri_image.get_fdata()
    mri_image_data = (mri_image_data - np.min(mri_image_data)) / (np.max(mri_image_data) - np.min(mri_image_data))
    mri_image_tensor = torch.tensor(mri_image_data, dtype=torch.float32)
    return mri_image_tensor

class MRIDataset(Dataset):
    def __init__(self, image_dir, mri_to_label, transform=None):
        self.image_dir = image_dir
        self.mri_to_label = mri_to_label
        self.image_ids = [image_id for image_id in mri_to_label.keys() if os.path.exists(os.path.join(image_dir, f"{image_id}.nii"))]
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        mri_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{mri_id}.nii")
        image_tensor = load_and_preprocess_mri(image_path)
        label = self.mri_to_label[mri_id]
        label_tensor = torch.tensor(label)

        # Apply augmentations based on class
        if label in [0, 2]:  # Assuming 0 and 2 are minority classes
            subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor.unsqueeze(0)))  # Add channel dimension
            subject = self.transform(subject)
            image_tensor = subject['image'].tensor.squeeze(0)  # Remove channel dimension after transform

        return image_tensor, label_tensor
    
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(UNet3D, self).__init__()
        # Reduced number of filters and simplified architecture
        self.enc1 = self.conv_block(in_channels, 16)  # Reduced from 32 filters
        self.enc2 = self.conv_block(16, 32)  # Reduced from 64 filters
        self.enc3 = self.conv_block(32, 64)  # Reduced from 128 filters
        self.enc4 = self.conv_block(64, 128)  # Reduced from 256 filters

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(128, 256)  # Bottleneck with reduced filters

        self.upconv4 = self.up_conv(256, 128)
        self.dec4 = self.conv_block(256, 128)
        self.upconv3 = self.up_conv(128, 64)
        self.dec3 = self.conv_block(128, 64)
        self.upconv2 = self.up_conv(64, 32)
        self.dec2 = self.conv_block(64, 32)
        self.upconv1 = self.up_conv(32, 16)
        self.dec1 = self.conv_block(32, 16)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16, out_channels)  # Reduced final layer complexity

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = self.pad_and_concat(dec4, enc4)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self.pad_and_concat(dec3, enc3)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self.pad_and_concat(dec2, enc2)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.pad_and_concat(dec1, enc1)
        dec1 = self.dec1(dec1)

        pooled = self.global_pool(dec1)
        flattened = pooled.view(pooled.size(0), -1)
        out = self.fc(flattened)

        return out

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.2)  # Added dropout to deeper layers
        )

    @staticmethod
    def up_conv(in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    @staticmethod
    def pad_and_concat(dec_tensor, enc_tensor):
        # Existing padding logic remains the same
        diffZ = enc_tensor.size(2) - dec_tensor.size(2)
        diffY = enc_tensor.size(3) - dec_tensor.size(3)
        diffX = enc_tensor.size(4) - dec_tensor.size(4)
        dec_tensor_padded = F.pad(dec_tensor, [
            diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2,
            diffZ // 2, diffZ - diffZ // 2
        ])
        return torch.cat((dec_tensor_padded, enc_tensor), dim=1)    
    
# Function to save the trained model
def save_model(model, path="unet3d_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the trained model
def load_model(path="unet3d_model.pth"):
    model = UNet3D(in_channels=1, out_channels=3)  # Initialize a new instance of the model
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

def collate_fn(batch):
    # Filter out any items that are None
    batch = [b for b in batch if b[0] is not None and b[1] is not None]

    if len(batch) == 0:
        return None, None  # Return None if the entire batch is invalid

    images, labels = zip(*batch)

    # Find the maximum dimensions in the batch
    max_depth = max([img.shape[0] for img in images])
    max_height = max([img.shape[1] for img in images])
    max_width = max([img.shape[2] for img in images])

    # Pad each image in the batch to the maximum dimensions
    padded_images = []
    for img in images:
        depth_diff = max_depth - img.shape[0]
        height_diff = max_height - img.shape[1]
        width_diff = max_width - img.shape[2]

        # Apply padding along each dimension (depth, height, width)
        padded_img = F.pad(img, (0, width_diff, 0, height_diff, 0, depth_diff))  # Pads along (W, H, D)
        padded_images.append(padded_img)

    # Stack the images and labels to form a batch
    images_tensor = torch.stack(padded_images)
    labels_tensor = torch.stack(labels)

    return images_tensor, labels_tensor

# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient computation during inference
        for images, labels in dataloader:
            images = images.unsqueeze(1).to(device)  # Add channel dimension and move to GPU
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the predicted class (highest score)
            
            all_preds.extend(preds.cpu().numpy())  # Collect the predictions
            all_labels.extend(labels.cpu().numpy())  # Collect the ground truth labels
    
    return all_preds, all_labels

# Calculate metrics from predictions and labels
def calculate_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=1)
    recall = recall_score(labels, preds, average='weighted', zero_division=1)
    f1 = f1_score(labels, preds, average='weighted', zero_division=1)
    cm = confusion_matrix(labels, preds)
    class_report = classification_report(labels, preds)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(class_report)

# Function to create dataloader
def create_dataloader(image_dir, label_file_path, batch_size=2):
    df_labels = pd.read_csv(label_file_path)
    mri_to_label = {row['MRI_ID']: row['Group'] for _, row in df_labels.iterrows()}
    
    # Instantiate the dataset with transformations
    dataset = MRIDataset(image_dir, mri_to_label, transform=augmentation_pipeline)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


# Function to calculate class weights
def calculate_class_weights(class_counts):
    total_samples = sum(class_counts.values())
    return {label: total_samples / count for label, count in class_counts.items()}

# Initialize the gradient scaler for mixed precision

# Add this function anywhere before you use it
def check_class_distribution(dataloader):
    class_counts = {0: 0, 1: 0, 2: 0}  # Assuming 3 classes: CN (0), MCI (1), AD (2)
    
    for images, labels in dataloader:
        unique, counts = torch.unique(labels, return_counts=True)
        for u, c in zip(unique.cpu().numpy(), counts.cpu().numpy()):
            class_counts[int(u)] += c
    return class_counts

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # You might want to increase alpha for minority classes.
        self.gamma = gamma  # Increasing gamma makes the loss focus more on hard-to-classify examples.
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
    def __init__(self, alpha=1, gamma=2, weights=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weights)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

scaler = GradScaler(device='cuda')

def train_model(model, dataloader, criterion, optimizer, device, num_epochs=10, max_grad_norm=1.0):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        epoch_preds = []
        epoch_labels = []
        
        for images, labels in dataloader:
            images = images.unsqueeze(1).to(device)  # Add channel dimension (1) and move to the correct device
            labels = labels.to(device)  # Move labels to the correct device

            optimizer.zero_grad()  # Zero the gradients

            # Mixed precision forward pass (optional)
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Clip gradients

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # Get predictions
            _, preds = torch.max(outputs, 1)
            epoch_preds.extend(preds.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

        if (epoch % 5) == 0:
            # Print epoch loss and class predictions
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")
            print("Predicted class distribution in the current epoch:")
            print(np.bincount(epoch_preds, minlength=3))  # Assuming 3 classes (0, 1, 2)
            print("True class distribution in the current epoch:")
            print(np.bincount(epoch_labels, minlength=3))

        torch.cuda.empty_cache()  # Free up GPU memory after each epoch


if __name__ == "__main__":
    # Define paths
    image_dir = "/db/proj-mag/segmented_1"
    csv_path = "/db/proj-mag/Label_File.csv"
    
    # Create the dataloader
    dataloader = create_dataloader(image_dir, csv_path)

    # Check class distribution and calculate class weights
    class_counts = check_class_distribution(dataloader)
    class_weights = calculate_class_weights(class_counts)
    weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32)

    # Set device and instantiate model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_channels=1, out_channels=3)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Define the loss function and optimizer
    weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights)], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=1, gamma=2, weights=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 30
    train_model(model, dataloader, criterion, optimizer, device, num_epochs)

    # Save the trained model
    save_model(model, path=r"unet3d_model_1.pth")
 
    # Evaluate the model
    preds, labels = evaluate_model(model, dataloader, device)
    calculate_metrics(preds, labels)

    # Optional: Inspect predictions on a small batch
    images, labels = next(iter(dataloader))
    images, labels = images.unsqueeze(1).to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        print("Predictions:", preds.cpu().numpy())
        print("Actual Labels:", labels.cpu().numpy())