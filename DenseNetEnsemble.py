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

lighter_transforms = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2)),  # Flip images randomly across all axes
    tio.RescaleIntensity((-1, 1)),   # Normalize intensity
    tio.RandomAffine(scales=(0.95, 1.05), degrees=5, translation=2),  # Less aggressive affine transformations
    tio.OneOf({
        tio.RandomNoise(p=0.1): 0.6,  # Lower impact and probability
        tio.RandomBiasField(p=0.1): 0.4
    })
])

# Function to load and normalize MRI images
def load_and_preprocess_mri(image_path):
    mri_image = nib.load(image_path)
    mri_image_data = mri_image.get_fdata()
    mri_image_data = (mri_image_data - np.min(mri_image_data)) / (np.max(mri_image_data) - np.min(mri_image_data))
    mri_image_tensor = torch.tensor(mri_image_data, dtype=torch.float32)
    return mri_image_tensor

class MRIDataset(Dataset):
    def __init__(self, image_dir, mri_to_label, transform=None, light_transform=None):
        self.image_dir = image_dir
        self.mri_to_label = mri_to_label
        self.image_ids = [image_id for image_id in mri_to_label.keys() if os.path.exists(os.path.join(image_dir, f"{image_id}.nii.gz"))]
        self.transform = transform
        self.light_transform = light_transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        mri_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{mri_id}.nii.gz")
        image_tensor = load_and_preprocess_mri(image_path)
        label = self.mri_to_label[mri_id]
        label_tensor = torch.tensor(label)

        # Apply augmentations based on class
        if label in [0, 2]:  # Assuming 0 and 2 are minority classes
            subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor.unsqueeze(0)))  # Add channel dimension
            subject = self.transform(subject)
            image_tensor = subject['image'].tensor.squeeze(0)  # Remove channel dimension after transform
            
        # if label in [1]:  # Assuming 0 and 2 are majority classes
        #     subject = tio.Subject(image=tio.ScalarImage(tensor=image_tensor.unsqueeze(0)))  # Add channel dimension
        #     subject = self.light_transform(subject)
        #     image_tensor = subject['image'].tensor.squeeze(0)  # Remove channel dimension after transform

        return image_tensor, label_tensor

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class DenseNet3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=3):
        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer (classifier)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))  # Global average pooling
        out = torch.flatten(out, 1)
        # Return features instead of class predictions
        return out

# Function to save the trained model
def save_model(model, path="DenseNet3D_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# Function to load the trained model
def load_model(path="DenseNet3D_model.pth"):
    model = DenseNet3D()  # Initialize a new instance of the model
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
    try:
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
        
        return acc, precision, recall, f1
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 0, 0, 0, 0  # Return zero or a default value for all metrics
    
# Function to create dataloader
def create_dataloader(image_dir, label_file_path, batch_size=10):
    df_labels = pd.read_csv(label_file_path)
    mri_to_label = {row['MRI_ID']: row['Group'] for _, row in df_labels.iterrows()}
    
    # Instantiate the dataset with transformations
    dataset = MRIDataset(image_dir, mri_to_label, transform=augmentation_pipeline, light_transform=lighter_transforms)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    return dataloader

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
        self.alpha = alpha
        self.gamma = gamma
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
        
def run_experiments(dataloader, device, alphas, gammas, num_epochs=10):
    results = {}
    for alpha in alphas:
        for gamma in gammas:
            model = DenseNet3D().to(device)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            print(f"Training with alpha: {alpha}, gamma: {gamma}")
            train_model(model, dataloader, optimizer, device, alpha, gamma, num_epochs)
            preds, labels = evaluate_model(model, dataloader, device)
            acc, prec, recall, f1 = calculate_metrics(preds, labels)
            results[(alpha, gamma)] = (acc, prec, recall, f1)
            save_model(model, 'dense_model_features.pth')
    return results

scaler = GradScaler(device='cuda')

def train_model(model, dataloader, optimizer, device, alpha, gamma, num_epochs=10, max_grad_norm=1.0):
    criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean').to(device)
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
    image_dir = "/db/proj-mag/final"
    csv_path = "/db/proj-mag/Label_File.csv"
    dataloader = create_dataloader(image_dir, csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alphas = [1.2]
    gammas = [4]
    results = run_experiments(dataloader, device, alphas, gammas, num_epochs=10)
    
    # Print the results for each combination
    for params, metrics in results.items():
        print(f"Alpha: {params[0]}, Gamma: {params[1]} - Metrics: {metrics}")