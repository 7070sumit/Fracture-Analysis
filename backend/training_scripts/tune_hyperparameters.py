import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging
import datetime

from model_manager import BoneFractureModel, FractureDataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def objective(trial):
    base_dir = Path(__file__).parent
    data_dir = base_dir.parent / "datasets" / "OAI_Processed"
    clinical_data_path = base_dir.parent / "datasets" / "Clinical_data" / "clinical_pool.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    
    # 1. Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.7)
    mlp_hidden_dim = trial.suggest_categorical("mlp_hidden_dim", [64, 128, 256])
    attention_dim = trial.suggest_categorical("attention_dim", [128, 256, 512])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # 2. Dataset Setup
    from torchvision import transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # We do a fast train/val split for tuning to save time
    train_base = FractureDataset(data_dir, split="train", clinical_data_path=clinical_data_path)
    val_base = FractureDataset(data_dir, split="val", clinical_data_path=clinical_data_path)
    all_images = train_base.images + val_base.images
    all_labels = train_base.labels + val_base.labels

    if not all_images:
        logger.error("No data found for tuning.")
        raise optuna.TrialPruned()

    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    train_dataset = FractureDataset(data_dir, split="train", transform=train_transform, clinical_data_path=clinical_data_path, images=train_imgs, labels=train_lbls)
    val_dataset = FractureDataset(data_dir, split="val", transform=val_transform, clinical_data_path=clinical_data_path, images=val_imgs, labels=val_lbls)

    # Sampler
    t_normal_count = sum(1 for label in train_dataset.labels if label == 0)
    t_fracture_count = len(train_dataset.labels) - t_normal_count
    t_class_weights = [1.0 / t_normal_count if t_normal_count > 0 else 0, 1.0 / t_fracture_count if t_fracture_count > 0 else 0]
    sample_weights = [t_class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, num_workers=2 if str(device) != 'mps' else 0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2 if str(device) != 'mps' else 0)

    # 3. Model Initialization
    model = BoneFractureModel(
        num_classes=2, 
        num_clinical_features=5, 
        dropout_rate=dropout_rate, 
        mlp_hidden_dim=mlp_hidden_dim, 
        attention_dim=attention_dim
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.clinical_mlp.parameters()) +
        list(model.attention.parameters()) +
        list(model.classifier.parameters())
    )
    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params, 'lr': lr}
    ], weight_decay=weight_decay)

    # Fast tuning loop (e.g. 3 epochs)
    epochs = 3
    
    for epoch in range(epochs):
        model.train()
        for (images, clinical), labels in train_loader:
            images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, clinical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        all_probs = []
        all_labels_val = []
        with torch.no_grad():
            for (images, clinical), labels in val_loader:
                images, clinical = images.to(device), clinical.to(device)
                outputs = model(images, clinical)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels_val.extend(labels.numpy())
                
        try:
            roc_auc = roc_auc_score(all_labels_val, all_probs)
        except ValueError:
            roc_auc = 0.0
            
        trial.report(roc_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return roc_auc

def main():
    logger.info("Starting Hyperparameter Tuning with Optuna")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5) # Reduced trials for demo/time constraints

    logger.info("Number of finished trials: {}".format(len(study.trials)))
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # Save best parameters
    base_dir = Path(__file__).parent
    models_dir = base_dir.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "best_hyperparameters.json", "w") as f:
        json.dump(trial.params, f, indent=4)
    logger.info(f"Saved best hyperparameters to {models_dir / 'best_hyperparameters.json'}")

if __name__ == "__main__":
    main()
