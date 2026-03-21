import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Optional, List
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import base64
from io import BytesIO
import cv2

from grad_cam import GradCAM, create_grad_cam_visualization

logger = logging.getLogger(__name__)

class BoneFractureModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Use ResNet50 with pretrained weights
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class FractureDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = Path(data_dir)
        # MURA uses 'valid' instead of 'val'
        self.split_name = "valid" if split == "val" else split
        self.split_dir = self.data_dir / self.split_name
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        csv_path = self.data_dir / f"{self.split_name}_image_paths.csv"
        
        if csv_path.exists():
            # MURA Dataset parsing
            with open(csv_path, 'r') as f:
                for line in f:
                    img_path_str = line.strip()
                    if not img_path_str: continue
                    
                    # paths in MURA csv start with "MURA-v1.1/", e.g., MURA-v1.1/train/XR_SHOULDER/...
                    if self.data_dir.name == "MURA-v1.1":
                        full_path = self.data_dir.parent / img_path_str
                    else:
                        full_path = self.data_dir / img_path_str.split("MURA-v1.1/")[-1]
                        
                    if full_path.exists():
                        self.images.append(str(full_path))
                        # "positive" means abnormal/fracture (1), "negative" means normal (0)
                        label = 1 if 'positive' in str(full_path) else 0
                        self.labels.append(label)
        else:
            # Handle standard custom dataset structure
            # Load normal images
            normal_dir = self.split_dir / "normal"
            if normal_dir.exists():
                for img_path in normal_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.images.append(str(img_path))
                        self.labels.append(0)
            
            # Load fractured images
            fractured_dir = self.split_dir / "fractured"
            if fractured_dir.exists():
                for img_path in fractured_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.images.append(str(img_path))
                        self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ModelManager:
    def __init__(self, model_dir=None, data_dir=None):
        base_dir = Path(__file__).parent
        self.model_dir = Path(model_dir) if model_dir else base_dir / "models"
        self.data_dir = Path(data_dir) if data_dir else base_dir / "data"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = BoneFractureModel(num_classes=2).to(self.device)
        self.grad_cam = None
        self.class_names = ["Normal", "Fracture"]
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model if exists
        self.load_model()
    
    def load_model(self):
        """Load trained model if exists"""
        model_path = self.model_dir / "best_model.pth"
        if model_path.exists():
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                logger.info(f"Model loaded from {model_path}")
                
                # Initialize Grad-CAM
                self.grad_cam = GradCAM(self.model, self.model.backbone.layer4)
                logger.info("Grad-CAM initialized")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("No trained model found. Please train a model first.")
    
    def predict(self, image_path: str, return_cam: bool = True) -> Dict:
        """Make prediction on a single image"""
        if self.grad_cam is None:
            # Create a simple prediction without trained model
            return {
                "prediction": "Model not trained",
                "confidence": 0.0,
                "probabilities": [0.5, 0.5],
                "class_idx": 0,
                "grad_cam_image": None
            }
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1)
        
        confidence, class_idx = probs.max(dim=1)
        confidence = confidence.item()
        class_idx = class_idx.item()
        
        result = {
            'prediction': self.class_names[class_idx],
            'confidence': round(confidence, 4),
            'class_idx': class_idx,
            'probabilities': [round(p, 4) for p in probs[0].cpu().numpy().tolist()]
        }
        
        # Generate Grad-CAM
        if return_cam and self.grad_cam:
            try:
                grad_cam_img = create_grad_cam_visualization(
                    self.model, 
                    self.grad_cam, 
                    image_tensor, 
                    image, 
                    class_idx
                )
                result['grad_cam_image'] = grad_cam_img
            except Exception as e:
                logger.error(f"Grad-CAM generation error: {e}")
                result['grad_cam_image'] = None
        
        return result
    
    def train_model(self, epochs: int = 10, batch_size: int = 16, 
                   learning_rate: float = 0.0001, training_state: Dict = None):
        """Train the model"""
        try:
            if training_state:
                training_state["status"] = "preparing_data"
            
            # Data transforms
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Create datasets
            train_dataset = FractureDataset(self.data_dir, split="train", transform=train_transform)
            val_dataset = FractureDataset(self.data_dir, split="val", transform=val_transform)
            
            if len(train_dataset) == 0:
                raise Exception("Training dataset is empty. Please add images to data/train folder.")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, #verbose=True
            )
            
            best_accuracy = 0
            history = {'train_loss': [], 'val_acc': []}
            
            if training_state:
                training_state["status"] = "training"
            
            import time
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                total_batches = len(train_loader)
                
                for batch_idx, (images, labels) in enumerate(train_loader):
                    batch_start_time = time.time()
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    if (batch_idx + 1) % 50 == 0:
                        batch_time = time.time() - batch_start_time
                        eta_epoch_mins = (total_batches - (batch_idx + 1)) * batch_time / 60
                        logger.info(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}, ETA for epoch: {eta_epoch_mins:.1f}m")
                
                train_loss /= total_batches
                history['train_loss'].append(train_loss)
                
                # Validation phase
                self.model.eval()
                correct = total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                val_accuracy = correct / total if total > 0 else 0
                history['val_acc'].append(val_accuracy)
                scheduler.step(val_accuracy)
                
                # Update training state
                if training_state:
                    training_state["current_epoch"] = epoch + 1
                    training_state["progress"] = ((epoch + 1) / epochs) * 100
                    training_state["train_loss"] = train_loss
                    training_state["val_accuracy"] = val_accuracy
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                
                # Save best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'accuracy': val_accuracy
                    }, self.model_dir / "best_model.pth")
                    logger.info(f"Best model saved (Acc: {best_accuracy:.4f})")
            
            # Save training history
            with open(self.model_dir / "training_history.json", "w") as f:
                json.dump(history, f)
            
            # Reload model and initialize Grad-CAM
            self.load_model()
            
            if training_state:
                training_state["is_training"] = False
                training_state["status"] = "completed"
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            if training_state:
                training_state["is_training"] = False
                training_state["status"] = f"error: {str(e)}"
            raise
    
    def get_metrics(self) -> Optional[Dict]:
        """Calculate model metrics on validation set"""
        model_path = self.model_dir / "best_model.pth"
        if not model_path.exists():
            return None
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        val_dataset = FractureDataset(self.data_dir, split="val", transform=val_transform)
        if len(val_dataset) == 0:
            return {"error": "No validation data available"}
        
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": conf_matrix
        }
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        train_dataset = FractureDataset(self.data_dir, split="train")
        val_dataset = FractureDataset(self.data_dir, split="val")
        
        train_normal = sum(1 for label in train_dataset.labels if label == 0)
        train_fractured = sum(1 for label in train_dataset.labels if label == 1)
        val_normal = sum(1 for label in val_dataset.labels if label == 0)
        val_fractured = sum(1 for label in val_dataset.labels if label == 1)
        
        return {
            "train": {
                "total": len(train_dataset),
                "normal": train_normal,
                "fractured": train_fractured
            },
            "val": {
                "total": len(val_dataset),
                "normal": val_normal,
                "fractured": val_fractured
            }
        }
