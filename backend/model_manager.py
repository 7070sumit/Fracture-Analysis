import time
import json
import base64
import logging
import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt

from grad_cam import GradCAM, create_grad_cam_visualization
from shap_explainer import create_shap_visualization

logger = logging.getLogger(__name__)

class BoneFractureModel(nn.Module):
    """
    Multimodal Fusion Architecture (ResNet50 + Clinical MLP).
    This model fuses raw X-ray image data with tabular clinical features (Age, BMI, etc.)
    using a Self-Attention mechanism to improve fracture risk prediction accuracy.
    """
    def __init__(self, num_classes=2, num_clinical_features=5, dropout_rate=0.5, mlp_hidden_dim=128, attention_dim=256):
        super().__init__()
        # ── ImageNet ResNet-50 Backbone (Visual Features) ────────────────────
        # Extracts spatial features from the X-ray image (outputs 2048-dim vector)
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove the default 1000-class ImageNet head

        # ── Clinical Feature MLP (Text/Data Features) ───────────────────────
        # Processes the 5 clinical data points (Age, BMI, Sex, Diabetes, Falls)
        # into a higher-dimensional embedding (outputs 128-dim vector)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.4), # Scaled dropout to prevent overfitting
            nn.Linear(64, mlp_hidden_dim),
            nn.ReLU(inplace=True)
        )

        # The combined dimension is Visual (2048) + Clinical (128) = 2176
        combined_dim = in_features + mlp_hidden_dim

        # ── Self-Attention Gate (Multimodal Fusion) ─────────────────────────
        # Learns which modality (Image vs Clinical) is more important for a specific patient.
        # It calculates a weight between 0 and 1 for every feature in the combined vector.
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, combined_dim),
            nn.Sigmoid()
        )

        # ── Final Classifier ────────────────────────────────────────────────
        # Takes the attention-weighted fused vector and predicts the final risk score
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, img, clinical=None):
        # 1. Extract Visual Features
        img_features = self.backbone(img)

        # 2. Handle missing clinical data gracefully (impute with zeros)
        if clinical is None:
            clinical = torch.zeros(img.size(0), 5, device=img.device)

        # 3. Extract Clinical Features
        clinical_features = self.clinical_mlp(clinical)

        # 4. Multimodal Fusion via Concatenation
        combined = torch.cat((img_features, clinical_features), dim=1)
        
        # 5. Apply Attention weights to the combined vector
        attn_weights = self.attention(combined)
        attended = combined * attn_weights

        # 6. Final classification logits
        return self.classifier(attended)


class FractureDataset(Dataset):
    def __init__(self, data_dir: str = None, split: str = "train", transform=None, clinical_dict: Optional[Dict] = None, clinical_data_path: Optional[str] = None, images=None, labels=None):
        if data_dir:
            self.data_dir = Path(data_dir)
            self.split_name = split
            self.split_dir = self.data_dir / self.split_name
        self.transform = transform
        
        # Prefer pre-loaded dictionary for speed
        if clinical_dict is not None:
            self.clinical_dict = clinical_dict
        else:
            self.clinical_dict = {}
            cpath = Path(clinical_data_path) if clinical_data_path else Path(__file__).parent / "datasets" / "Clinical_data" / "clinical_pool.csv"
            if cpath.exists():
                try:
                    df = pd.read_csv(cpath)
                    feature_cols = ['Age_std', 'BMI_std', 'P02SEX', 'V00DIAB', 'V00FALL']
                    if all(col in df.columns for col in feature_cols) and 'ID' in df.columns:
                        for _, row in df.iterrows():
                            self.clinical_dict[str(int(row['ID']))] = row[feature_cols].values.astype(np.float32)
                except Exception as e:
                    logger.error(f"Error loading clinical pool: {e}")
        
        if images is not None and labels is not None:
            self.images = images
            self.labels = labels
        else:
            self.images = []
            self.labels = []
            
            # Handle standard custom dataset structure
            if hasattr(self, 'split_dir'):
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
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # Extract Subject ID from filename (assuming format like '9002316_001.png')
        filename = Path(image_path).stem
        subj_id = filename.split('_')[0] if '_' in filename else filename
        
        # Use exact mapping to real clinical profiles
        if self.clinical_dict and subj_id in self.clinical_dict:
            clinical_features = torch.from_numpy(self.clinical_dict[subj_id])
        else:
            # Fallback to zero features if ID not found or dict empty
            clinical_features = torch.zeros(5, dtype=torch.float32)
        
        return (image, clinical_features), label

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
        
        self.model = BoneFractureModel(num_classes=2, num_clinical_features=5).to(self.device)
        self.grad_cam = None
        self.class_names = ["Normal", "Fracture"]
        
        # ── PERFORMANCE OPTIMIZATION: In-Memory Clinical Data Caching ──
        # Instead of doing slow Disk I/O (reading the CSV file via Pandas) every single time 
        # a prediction is requested, we load all clinical profiles into RAM instantly 
        # when the server starts. This reduces prediction latency by ~50ms per request.
        self.clinical_data_path = base_dir / "datasets" / "Clinical_data" / "clinical_pool.csv"
        self.clinical_dict = {}
        if self.clinical_data_path.exists():
            try:
                df = pd.read_csv(self.clinical_data_path)
                feature_cols = ['Age_std', 'BMI_std', 'P02SEX', 'V00DIAB', 'V00FALL']
                if all(col in df.columns for col in feature_cols) and 'ID' in df.columns:
                    for _, row in df.iterrows():
                        self.clinical_dict[str(int(row['ID']))] = row[feature_cols].values.astype(np.float32).tolist()
                logger.info(f"Cached {len(self.clinical_dict)} clinical profiles in memory.")
            except Exception as e:
                logger.error(f"Error caching clinical pool: {e}")
        
        # Image preprocessing with Data Augmentation
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
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
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.eval()
                logger.info(f"Model loaded from {model_path}")
                
                # Initialize Grad-CAM
                self.grad_cam = GradCAM(self.model, self.model.backbone.layer4)
                logger.info("Grad-CAM initialized")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        else:
            logger.warning("No trained model found. Please train a model first.")
    
    def predict(self, image_path: str, return_cam: bool = True, clinical_features: Optional[List[float]] = None, original_filename: str = None) -> Dict:
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
        image = ImageOps.exif_transpose(image)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        clin_tensor = None
        if clinical_features is not None:
            clin_tensor = torch.tensor([clinical_features], dtype=torch.float32).to(self.device)
        else:
            # Map filename to pre-loaded clinical dict
            filename = Path(original_filename).stem if original_filename else Path(image_path).stem
            subj_id = filename.split('_')[0] if '_' in filename else filename
            
            if subj_id in self.clinical_dict:
                mapped_features = self.clinical_dict[subj_id]
                clin_tensor = torch.tensor([mapped_features], dtype=torch.float32).to(self.device)
                logger.info(f"Mapped prediction image to clinical profile {subj_id}")
            else:
                logger.warning(f"Could not find clinical profile for ID {subj_id}")
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image_tensor, clin_tensor)
            probs = torch.softmax(logits, dim=1)
        
        confidence, class_idx = probs.max(dim=1)
        confidence = confidence.item()
        class_idx = class_idx.item()
        
        result = {
            'prediction': self.class_names[class_idx],
            'confidence': float(confidence),
            'class_idx': class_idx,
            'probabilities': [float(p) for p in probs[0].cpu().numpy().tolist()]
        }
        
        # Generate Grad-CAM
        if return_cam and self.grad_cam:
            try:
                grad_cam_img = create_grad_cam_visualization(
                    self.model, 
                    self.grad_cam, 
                    image_tensor, 
                    image, 
                    1, # Always target 'Fracture' class (class index 1)
                    clin_tensor
                )
                result['grad_cam_image'] = grad_cam_img
            except Exception as e:
                logger.error(f"Grad-CAM generation error: {e}")
                result['grad_cam_image'] = None
                
            # Generate SHAP for clinical features
            if clin_tensor is not None:
                try:
                    shap_img = create_shap_visualization(
                        self.model,
                        image_tensor,
                        clin_tensor
                    )
                    result['shap_image'] = shap_img
                except Exception as e:
                    logger.error(f"SHAP generation error: {e}")
                    result['shap_image'] = None
        
        return result
    
    def train_model(self, epochs: int = 10, batch_size: int = 16, 
                   learning_rate: float = 0.0001, training_state: Dict = None):
        """Train the model"""
        try:
            if training_state:
                training_state["status"] = "preparing_data"
            

            
            # Create datasets using pre-loaded clinical dictionary for extreme speed
            train_base = FractureDataset(self.data_dir, split="train", clinical_dict=self.clinical_dict)
            val_base   = FractureDataset(self.data_dir, split="val", clinical_dict=self.clinical_dict)
            all_images = train_base.images + val_base.images
            all_labels = train_base.labels + val_base.labels
            
            train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
                all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
            )
            
            train_dataset = FractureDataset(self.data_dir, split="train", transform=self.train_transform, clinical_dict=self.clinical_dict, images=train_imgs, labels=train_lbls)
            val_dataset   = FractureDataset(self.data_dir, split="val", transform=self.transform, clinical_dict=self.clinical_dict, images=val_imgs, labels=val_lbls)
            
            if len(train_dataset) == 0:
                raise Exception("Training dataset is empty.")
            
            # Compute class weights for WeightedRandomSampler
            t_normal_count = sum(1 for label in train_dataset.labels if label == 0)
            t_fracture_count = len(train_dataset.labels) - t_normal_count
            
            t_class_weights = [1.0 / t_normal_count if t_normal_count > 0 else 0, 
                               1.0 / t_fracture_count if t_fracture_count > 0 else 0]
            sample_weights = [t_class_weights[label] for label in train_dataset.labels]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            
            # Experiment Tracking Setup
            run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = self.model_dir / f"run_{run_timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Experiment Tracking: Saving to {run_dir}")
            
            # Training setup
            # Label Smoothing: prevents overconfidence, improves generalization (used by Zhang 2025, AUC 0.949)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            
            # Differential Learning Rates: backbone learns slowly (fine-tuning X-ray patterns),
            # classifier head learns fast (new task-specific weights)
            backbone_params = list(self.model.backbone.parameters())
            head_params = (
                list(self.model.clinical_mlp.parameters()) +
                list(self.model.attention.parameters()) +
                list(self.model.classifier.parameters())
            )
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': learning_rate * 0.1},  # 1e-5 for backbone
                {'params': head_params, 'lr': learning_rate}              # 1e-4 for head
            ], weight_decay=1e-4)
            # Cosine Annealing: smoothly decays LR — more stable than ReduceLROnPlateau
            # avoids the hard LR cuts that caused our plateaus in earlier runs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
            
            best_roc_auc  = 0.0
            best_accuracy = 0.0
            history = {'train_loss': [], 'val_acc': []}
            
            if training_state:
                training_state["status"] = "training"
            
            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                total_batches = len(train_loader)
                
                for batch_idx, ((images, clinical), labels) in enumerate(train_loader):
                    batch_start_time = time.time()
                    images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images, clinical)
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
                
                # Validation phase — using Balanced Accuracy (immune to class imbalance)
                self.model.eval()
                tp = tn = fp = fn = 0
                correct = total = 0
                all_probs = []
                all_labels_val = []

                with torch.no_grad():
                    for (images, clinical), labels in val_loader:
                        images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                        outputs = self.model(images, clinical)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        # Per-class tracking for balanced accuracy
                        for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                            if true == 1 and pred == 1:
                                tp += 1
                            elif true == 0 and pred == 0:
                                tn += 1
                            elif true == 0 and pred == 1:
                                fp += 1
                            elif true == 1 and pred == 0:
                                fn += 1

                        # Collect probabilities for ROC-AUC
                        probs = torch.softmax(outputs, dim=1)[:, 1]  # fracture probability
                        all_probs.extend(probs.cpu().numpy())
                        all_labels_val.extend(labels.cpu().numpy())

                val_accuracy = correct / total if total > 0 else 0
                sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
                balanced_acc = (sensitivity + specificity) / 2
                # ROC-AUC — primary research metric (beating Panfilov's 0.70-0.76 on OAI)
                try:
                    roc_auc = roc_auc_score(all_labels_val, all_probs)
                except Exception:
                    roc_auc = 0.0

                history['val_acc'].append(balanced_acc)
                history.setdefault('val_raw_acc', []).append(val_accuracy)
                history.setdefault('sensitivity', []).append(sensitivity)
                history.setdefault('specificity', []).append(specificity)
                history.setdefault('roc_auc', []).append(roc_auc)
                scheduler.step()

                # Update training state
                if training_state:
                    training_state["current_epoch"] = epoch + 1
                    training_state["progress"] = ((epoch + 1) / epochs) * 100
                    training_state["train_loss"] = train_loss
                    training_state["val_accuracy"] = balanced_acc

                logger.info(
                    f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} | "
                    f"ROC-AUC: {roc_auc:.4f} | "
                    f"Balanced Acc: {balanced_acc*100:.1f}% | "
                    f"Sensitivity: {sensitivity*100:.1f}% | "
                    f"Specificity: {specificity*100:.1f}%"
                )

                # Save best model based on ROC-AUC (primary metric vs Panfilov 0.70-0.76 benchmark)
                if roc_auc > best_roc_auc:
                    best_roc_auc  = roc_auc
                    best_accuracy = balanced_acc
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'roc_auc': roc_auc,
                        'balanced_accuracy': balanced_acc,
                        'sensitivity': sensitivity,
                        'specificity': specificity,
                        'raw_accuracy': val_accuracy
                    }, run_dir / "best_model.pth")
                    torch.save(self.model.state_dict(), self.model_dir / "best_model.pth")
                    logger.info(
                        f"✓ Best model saved → ROC-AUC: {best_roc_auc:.4f} | "
                        f"Balanced Acc: {balanced_acc*100:.1f}% | Sensitivity: {sensitivity*100:.1f}%"
                    )
            
            # Save training history
            with open(run_dir / "training_history.json", "w") as f:
                json.dump(history, f)
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
        
        val_dataset = FractureDataset(self.data_dir, split="val", transform=self.transform, clinical_data_path=self.clinical_data_path)
        if len(val_dataset) == 0:
            return {"error": "No validation data available"}
        
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for (images, clinical), labels in val_loader:
                images, clinical = images.to(self.device), clinical.to(self.device)
                outputs = self.model(images, clinical)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy()) # Probs for positive class (Fracture)
                all_labels.extend(labels.numpy())
        
        # Calculate base metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds).tolist()
        
        # Advanced Calibration Metrics
        roc_auc = 0.0
        brier_score = 0.0
        roc_curve_image = None
        calibration_curve_image = None
        
        if len(set(all_labels)) > 1: # Require both classes present
            roc_auc = roc_auc_score(all_labels, all_probs)
            brier_score = brier_score_loss(all_labels, all_probs)
            
            # Generate ROC plot
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            plt.switch_backend('agg')
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color='#008bfb', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            roc_curve_image = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            
            # Generate Calibration plot
            fraction_of_positives, mean_predicted_value = calibration_curve(all_labels, all_probs, n_bins=10)
            plt.figure(figsize=(6, 5))
            plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='#ff0051', label='Model')
            plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            plt.ylabel("Fraction of positives")
            plt.xlabel("Mean predicted value")
            plt.title(f'Calibration Curve (Brier: {brier_score:.3f})')
            plt.legend(loc="lower right")
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            calibration_curve_image = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": conf_matrix,
            "roc_auc": round(roc_auc, 4),
            "brier_score": round(brier_score, 4),
            "roc_curve_image": roc_curve_image,
            "calibration_curve_image": calibration_curve_image
        }
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        train_dataset = FractureDataset(self.data_dir, split="train", clinical_data_path=self.clinical_data_path)
        val_dataset = FractureDataset(self.data_dir, split="val", clinical_data_path=self.clinical_data_path)
        
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

    def run_kfold_cv(self, k_folds: int = 5, epochs_per_fold: int = 5, batch_size: int = 16, learning_rate: float = 0.0001):
        """Run K-Fold Cross Validation and aggregate results"""
        logger.info(f"Starting {k_folds}-Fold Cross Validation")
        
        # Experiment Tracking Setup
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = self.model_dir / f"kfold_{run_timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment Tracking: Saving to {run_dir}")
        
        # We need a combined dataset for K-Fold
        # Since FractureDataset currently loads by split, we'll load both and concat
        train_dataset = FractureDataset(data_dir=str(self.data_dir), split="train", clinical_data_path=self.clinical_data_path)
        val_dataset = FractureDataset(data_dir=str(self.data_dir), split="val", clinical_data_path=self.clinical_data_path)
        
        all_images = train_dataset.images + val_dataset.images
        all_labels = train_dataset.labels + val_dataset.labels
        
        if len(all_images) == 0:
            logger.error("Dataset is empty. Cannot run K-Fold CV.")
            return None
            
        # Class weights not needed as we will use WeightedRandomSampler
        logger.info(f"Using WeightedRandomSampler to balance the classes dynamically.")
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(all_images)):
            logger.info(f"--- Fold {fold+1}/{k_folds} ---")
            
            # Reset model weights for each fold
            self.model = BoneFractureModel(num_classes=2, num_clinical_features=5).to(self.device)
            criterion = nn.CrossEntropyLoss()
            backbone_params = list(self.model.backbone.parameters())
            head_params = (
                list(self.model.clinical_mlp.parameters()) +
                list(self.model.attention.parameters()) +
                list(self.model.classifier.parameters())
            )
            optimizer = torch.optim.Adam([
                {'params': backbone_params, 'lr': learning_rate * 0.1},
                {'params': head_params, 'lr': learning_rate}
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3
            )
            
            # Create fold-specific datasets with appropriate transforms
            fold_train_images = [all_images[i] for i in train_ids]
            fold_train_labels = [all_labels[i] for i in train_ids]
            fold_val_images = [all_images[i] for i in val_ids]
            fold_val_labels = [all_labels[i] for i in val_ids]
            
            fold_train_dataset = FractureDataset(
                transform=self.train_transform, clinical_data_path=self.clinical_data_path,
                images=fold_train_images, labels=fold_train_labels
            )
            fold_val_dataset = FractureDataset(
                transform=self.transform, clinical_data_path=self.clinical_data_path,
                images=fold_val_images, labels=fold_val_labels
            )
            
            # Compute fold-specific sampler weights
            f_normal_count = sum(1 for label in fold_train_labels if label == 0)
            f_fracture_count = len(fold_train_labels) - f_normal_count
            f_weights = [1.0 / f_normal_count if f_normal_count > 0 else 0,
                         1.0 / f_fracture_count if f_fracture_count > 0 else 0]
            f_sample_weights = [f_weights[label] for label in fold_train_labels]
            f_sampler = WeightedRandomSampler(weights=f_sample_weights, num_samples=len(fold_train_labels), replacement=True)
            
            train_loader = DataLoader(fold_train_dataset, batch_size=batch_size, sampler=f_sampler)
            val_loader = DataLoader(fold_val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train
            for epoch in range(epochs_per_fold):
                self.model.train()
                train_loss = 0
                n_batches  = 0
                for (images, clinical), labels in train_loader:
                    images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(images, clinical)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    n_batches  += 1

                train_loss /= max(n_batches, 1)

                # Epoch validation for scheduler
                self.model.eval()
                correct = total = 0
                with torch.no_grad():
                    for (images, clinical), labels in val_loader:
                        images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                        outputs = self.model(images, clinical)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_acc = correct / total if total > 0 else 0
                scheduler.step(val_acc)
                logger.info(f"  [Fold {fold+1}/{k_folds}] Epoch {epoch+1}/{epochs_per_fold} - Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.1f}%")

                    
            # Evaluate at the end of the fold
            self.model.eval()
            all_preds, all_probs, all_targets = [], [], []
            with torch.no_grad():
                for (images, clinical), labels in val_loader:
                    images, clinical, labels = images.to(self.device), clinical.to(self.device), labels.to(self.device)
                    outputs = self.model(images, clinical)
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(all_targets, all_preds)
            roc_auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
            brier = brier_score_loss(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
            
            logger.info(f"Fold {fold+1} Results - Acc: {accuracy:.4f}, AUC: {roc_auc:.4f}, Brier: {brier:.4f}")
            
            fold_results.append({
                "fold": fold + 1,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "brier_score": brier
            })
            
        # Aggregate results
        avg_acc = sum(r["accuracy"] for r in fold_results) / k_folds
        avg_auc = sum(r["roc_auc"] for r in fold_results) / k_folds
        avg_brier = sum(r["brier_score"] for r in fold_results) / k_folds
        
        final_results = {
            "k_folds": k_folds,
            "average_accuracy": round(avg_acc, 4),
            "average_roc_auc": round(avg_auc, 4),
            "average_brier_score": round(avg_brier, 4),
            "fold_details": fold_results
        }
        
        with open(run_dir / "kfold_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
        with open(self.model_dir / "kfold_results.json", "w") as f:
            json.dump(final_results, f, indent=4)
            
        logger.info(f"K-Fold CV Completed. Avg Acc: {avg_acc:.4f}, Avg AUC: {avg_auc:.4f}")
        return final_results
