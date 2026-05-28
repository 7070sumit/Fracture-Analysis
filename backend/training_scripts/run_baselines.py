import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import numpy as np
import json
from pathlib import Path
import logging

from model_manager import FractureDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Baseline 1: Unimodal Image Baseline (ResNet-50)
# ---------------------------------------------------------
class ImageOnlyBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, img, clinical=None):
        return self.backbone(img)

def run_image_baseline(all_images, all_labels, data_dir, clinical_data_path, device, k_folds=5, epochs=3):
    logger.info("Running Unimodal Image Baseline (ResNet-50)")
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(all_images)):
        logger.info(f"Image Baseline - Fold {fold+1}/{k_folds}")
        model = ImageOnlyBaseline().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        fold_train_images = [all_images[i] for i in train_ids]
        fold_train_labels = [all_labels[i] for i in train_ids]
        fold_val_images = [all_images[i] for i in val_ids]
        fold_val_labels = [all_labels[i] for i in val_ids]
        
        train_dataset = FractureDataset(transform=train_transform, clinical_data_path=clinical_data_path, images=fold_train_images, labels=fold_train_labels)
        val_dataset = FractureDataset(transform=val_transform, clinical_data_path=clinical_data_path, images=fold_val_images, labels=fold_val_labels)
        
        # Sampler
        t_normal = sum(1 for label in fold_train_labels if label == 0)
        t_frac = len(fold_train_labels) - t_normal
        t_weights = [1.0/t_normal if t_normal>0 else 0, 1.0/t_frac if t_frac>0 else 0]
        sample_weights = [t_weights[l] for l in fold_train_labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(fold_train_labels), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        for epoch in range(epochs):
            model.train()
            for (images, _), labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        model.eval()
        all_preds, all_probs, all_targets = [], [], []
        with torch.no_grad():
            for (images, _), labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                _, preds = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                
        acc = accuracy_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
        brier = brier_score_loss(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.0
        
        results.append({"fold": fold+1, "accuracy": acc, "roc_auc": auc, "brier_score": brier})
        logger.info(f"Fold {fold+1} - Acc: {acc:.4f}, AUC: {auc:.4f}")

    avg_acc = sum(r['accuracy'] for r in results)/k_folds
    avg_auc = sum(r['roc_auc'] for r in results)/k_folds
    return {"average_accuracy": avg_acc, "average_roc_auc": avg_auc, "folds": results}

# ---------------------------------------------------------
# Baseline 2: Unimodal Clinical Baseline (XGBoost)
# ---------------------------------------------------------
def run_clinical_baseline(all_images, all_labels, clinical_data_path, k_folds=5):
    logger.info("Running Unimodal Clinical Baseline (XGBoost)")
    
    # We need to extract just the clinical features
    dummy_dataset = FractureDataset(clinical_data_path=clinical_data_path, images=all_images, labels=all_labels)
    X_clinical = []
    y_clinical = []
    
    for i in range(len(dummy_dataset)):
        # _getitem_ returns ((image, clinical_features), label)
        # We just want clinical features
        # Note: image loading is slow, but we do it once for extraction
        _, clinical_features = dummy_dataset[i][0]
        label = dummy_dataset[i][1]
        X_clinical.append(clinical_features.numpy())
        y_clinical.append(label)
        
    X_clinical = np.array(X_clinical)
    y_clinical = np.array(y_clinical)
    
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_clinical)):
        X_train, X_val = X_clinical[train_ids], X_clinical[val_ids]
        y_train, y_val = y_clinical[train_ids], y_clinical[val_ids]
        
        # Calculate scale_pos_weight for imbalanced classes
        neg_count = sum(y_train == 0)
        pos_count = sum(y_train == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
        
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs) if len(set(y_val)) > 1 else 0.0
        brier = brier_score_loss(y_val, probs) if len(set(y_val)) > 1 else 0.0
        
        results.append({"fold": fold+1, "accuracy": acc, "roc_auc": auc, "brier_score": brier})
        logger.info(f"Fold {fold+1} - Acc: {acc:.4f}, AUC: {auc:.4f}")
        
    avg_acc = sum(r['accuracy'] for r in results)/k_folds
    avg_auc = sum(r['roc_auc'] for r in results)/k_folds
    return {"average_accuracy": avg_acc, "average_roc_auc": avg_auc, "folds": results}

def main():
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    clinical_data_path = base_dir / "datasets" / "Clinical_data" / "clinical_pool.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")

    # Load full dataset paths
    train_dataset = FractureDataset(data_dir=str(data_dir), split="train", clinical_data_path=clinical_data_path)
    val_dataset = FractureDataset(data_dir=str(data_dir), split="val", clinical_data_path=clinical_data_path)
    all_images = train_dataset.images + val_dataset.images
    all_labels = train_dataset.labels + val_dataset.labels

    if not all_images:
        logger.error("No data found for baseline evaluation.")
        return

    # Run Baselines
    # We use fewer epochs (3) for the image baseline here for demonstration/speed, 
    # but in a real paper it would be tuned fully.
    img_results = run_image_baseline(all_images, all_labels, data_dir, clinical_data_path, device, k_folds=3, epochs=3)
    clin_results = run_clinical_baseline(all_images, all_labels, clinical_data_path, k_folds=3)

    final_report = {
        "Image_Only_Baseline_ResNet50": img_results,
        "Clinical_Only_Baseline_XGBoost": clin_results
    }

    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "baseline_results.json", "w") as f:
        json.dump(final_report, f, indent=4)
        
    logger.info(f"Baseline results saved to {models_dir / 'baseline_results.json'}")

if __name__ == "__main__":
    main()
