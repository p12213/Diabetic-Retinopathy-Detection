"""
Diabetic Retinopathy PyTorch Model Training Script.
Calculates Quadratic Weighted Kappa (QWK), Accuracy, and saves best weights.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import get_model
from dataset import RetinopathyDataset, get_transforms

def compute_qwk(y_true, y_pred):
    """Calculates Quadratic Weighted Kappa (QWK) - Official APTOS evaluation metric."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_qwk = compute_qwk(all_targets, all_preds)
    
    return epoch_loss, epoch_acc, epoch_qwk

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_qwk = compute_qwk(all_targets, all_preds)
    
    return val_loss, val_acc, val_qwk, all_targets, all_preds

def main():
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Deep Learning Classifier")
    parser.add_argument("--csv_path", type=str, default="data/train.csv", help="Path to CSV annotations file")
    parser.add_argument("--img_dir", type=str, default="data/train_images", help="Path to fundus image directory")
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "efficientnet_b4"], help="Model backbone")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=512, help="Input image dimension")
    parser.add_argument("--save_path", type=str, default="checkpoints/best_dr_model.pth", help="Output checkpoint file path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Backbone: {args.backbone}")

    if not os.path.exists(args.csv_path):
        print(f"Error: CSV path '{args.csv_path}' not found. Please place APTOS train.csv or EyePACS annotations file there.")
        return

    df = pd.read_csv(args.csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
    print(f"Dataset split: {len(train_df)} train images, {len(val_df)} validation images")

    train_transform, val_transform = get_transforms(img_size=args.img_size)
    train_dataset = RetinopathyDataset(train_df, args.img_dir, transform=train_transform)
    val_dataset = RetinopathyDataset(val_df, args.img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model(backbone=args.backbone, num_classes=5, pretrained=True).to(device)
    
    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_qwk = -1.0

    print("\n--- Starting Training Loop ---")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc, train_qwk = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_qwk, val_targets, val_preds = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} QWK: {train_qwk:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_qwk': val_qwk,
                'val_acc': val_acc,
                'backbone': args.backbone
            }, args.save_path)
            print(f"  >>> Checkpoint Saved! Best Val QWK: {best_qwk:.4f}")

    print(f"\nTraining Complete. Best Validation QWK: {best_qwk:.4f}")

if __name__ == "__main__":
    main()
