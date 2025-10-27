import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import the fixed model implementations
from swin_transformer_fixed import SwinTransformerDeepfake
from diet_transformer_fixed import DietTransformerDeepfake

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load real images (label 0)
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            for img_name in os.listdir(real_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(real_dir, img_name))
                    self.labels.append(0)
        
        # Load fake images (label 1)
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            for img_name in os.listdir(fake_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(fake_dir, img_name))
                    self.labels.append(1)
        
        print(f"Loaded {len(self.data)} images: {self.labels.count(0)} real, {self.labels.count(1)} fake")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Training function
def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, model_name="model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_train / total_train:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * correct_train / total_train
        val_acc = 100 * correct_val / total_val
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name}_model.pth')
        
        scheduler.step()
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Plotting function
def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.title(f'{model_name} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot learning curve
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} - Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Evaluation function
def evaluate_model(model, test_loader, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification Report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=['Real', 'Fake'])
    print(f"\n{model_name} Classification Report:")
    print(report)
    
    return cm, report

# GradCAM visualization function
def visualize_gradcam(model, image_path, model_name, target_layers):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Preprocess for model
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Create GradCAM
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    
    # Generate CAM
    targets = [ClassifierOutputTarget(1)]  # Target fake class
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize original image to match model input
    image_resized = cv2.resize(image_np, (224, 224))
    image_resized = image_resized / 255.0
    
    # Create visualization
    visualization = show_cam_on_image(image_resized, grayscale_cam, use_rgb=True)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(grayscale_cam, cmap='hot')
    plt.title('GradCAM Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(visualization)
    plt.title('GradCAM Overlay')
    plt.axis('off')
    
    plt.suptitle(f'{model_name} - GradCAM Visualization')
    plt.tight_layout()
    plt.savefig(f'{model_name}_gradcam_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set dataset path
    DATASET_PATH = "dataset"  # Change this to your dataset folder path
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset folder '{DATASET_PATH}' not found!")
        print("Please ensure your dataset folder contains 'real' and 'fake' subfolders.")
        exit(1)
    
    # Create datasets
    full_dataset = DeepfakeDataset(DATASET_PATH, transform=transform_train)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create separate dataset instances for validation and test (with proper transforms)
    val_dataset_transformed = DeepfakeDataset(DATASET_PATH, transform=transform_test)
    test_dataset_transformed = DeepfakeDataset(DATASET_PATH, transform=transform_test)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Dataset splits: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Train Swin Transformer
    print("\n" + "="*50)
    print("Training Swin Transformer Model")
    print("="*50)
    
    swin_model = SwinTransformerDeepfake(num_classes=2)
    swin_train_losses, swin_val_losses, swin_train_acc, swin_val_acc = train_model(
        swin_model, train_loader, val_loader, num_epochs=20, model_name="SwinTransformer"
    )
    
    # Plot Swin Transformer results
    plot_training_history(swin_train_losses, swin_val_losses, swin_train_acc, swin_val_acc, "Swin Transformer")
    
    # Evaluate Swin Transformer
    swin_model.load_state_dict(torch.load('best_SwinTransformer_model.pth'))
    swin_cm, swin_report = evaluate_model(swin_model, test_loader, "Swin Transformer")
    
    # Train Diet Transformer
    print("\n" + "="*50)
    print("Training Diet Transformer Model")
    print("="*50)
    
    diet_model = DietTransformerDeepfake(num_classes=2)
    diet_train_losses, diet_val_losses, diet_train_acc, diet_val_acc = train_model(
        diet_model, train_loader, val_loader, num_epochs=20, model_name="DietTransformer"
    )
    
    # Plot Diet Transformer results
    plot_training_history(diet_train_losses, diet_val_losses, diet_train_acc, diet_val_acc, "Diet Transformer")
    
    # Evaluate Diet Transformer
    diet_model.load_state_dict(torch.load('best_DietTransformer_model.pth'))
    diet_cm, diet_report = evaluate_model(diet_model, test_loader, "Diet Transformer")
    
    # GradCAM visualization (you need to specify an image path)
    sample_image_path = None
    fake_dir = os.path.join(DATASET_PATH, 'fake')
    if os.path.exists(fake_dir):
        fake_images = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if fake_images:
            sample_image_path = os.path.join(fake_dir, fake_images[0])
    
    if sample_image_path and os.path.exists(sample_image_path):
        print("\nGenerating GradCAM visualizations...")
        
        # Swin Transformer GradCAM
        try:
            swin_target_layers = [swin_model.layers[-1].blocks[-1].norm1]
            visualize_gradcam(swin_model, sample_image_path, "Swin Transformer", swin_target_layers)
        except Exception as e:
            print(f"Error generating Swin Transformer GradCAM: {e}")
        
        # Diet Transformer GradCAM
        try:
            diet_target_layers = [diet_model.shared_blocks[-1].norm1]
            visualize_gradcam(diet_model, sample_image_path, "Diet Transformer", diet_target_layers)
        except Exception as e:
            print(f"Error generating Diet Transformer GradCAM: {e}")
    
    print("\nTraining and evaluation completed!")
    print("Model checkpoints saved as 'best_SwinTransformer_model.pth' and 'best_DietTransformer_model.pth'")
    print("Training plots and confusion matrices saved as PNG files.")