"""
train.py - Train PointNet++ on GT Fragments
=============================================
Automatically loads all GT fragments from data/with_gt/,
trains PointNet++, saves best model checkpoint.

Usage:
  python3 train.py
  python3 train.py --epochs 200 --batch_size 64 --gpu 0
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import config
from model.pointnet2 import PointNet2Classifier
from model.dataset import BreakSurfaceDataset, load_fragments


def train(args):
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    print(f"\n{'#'*70}")
    print(f"POINTNET++ TRAINING")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Points per sample: {args.num_points}")
    print(f"{'#'*70}")

    # ---- Load Data ----
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}")

    fragments, frag_names = load_fragments(config.DATA_GT_DIR, voxel_size=args.voxel)

    if not fragments:
        print("  No GT fragments found!")
        return

    # Create dataset
    print(f"\n  Creating dataset...")
    dataset = BreakSurfaceDataset(
        fragments,
        num_points=args.num_points,
        use_normals=True,
        augment=True,
        balance=True,
        samples_per_fragment=args.samples_per_frag)

    # Split train/val (80/20)
    n_total = len(dataset)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # Disable augmentation for validation
    # (random_split shares the underlying dataset, so we can't easily disable)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    print(f"  Train: {n_train:,} samples")
    print(f"  Val:   {n_val:,} samples")

    # ---- Model ----
    print(f"\n{'='*70}")
    print("MODEL")
    print(f"{'='*70}")

    model = PointNet2Classifier(
        input_channels=config.INPUT_CHANNELS,
        dropout=config.DROPOUT).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ---- Loss & Optimizer ----
    # Compute class weights from data
    all_labels = dataset.labels[dataset.indices]
    n_break = (all_labels == 1).sum()
    n_orig = (all_labels == 0).sum()
    weight_break = n_orig / max(n_break, 1)
    weight_orig = 1.0
    weights = torch.tensor([weight_orig, weight_break], dtype=torch.float32).to(device)

    print(f"  Class weights: original={weight_orig:.2f}, break={weight_break:.2f}")

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training Loop ----
    print(f"\n{'='*70}")
    print("TRAINING")
    print(f"{'='*70}")

    best_f1 = 0
    patience_counter = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [],
               'val_prec': [], 'val_rec': []}

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits, _ = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += data.size(0)

        scheduler.step()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validate ----
        model.eval()
        val_loss = 0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                logits, _ = model(data)
                loss = criterion(logits, target)

                val_loss += loss.item() * data.size(0)
                val_total += data.size(0)

                pred = logits.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= val_total
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        val_f1 = f1_score(all_targets, all_preds)
        val_prec = precision_score(all_targets, all_preds, zero_division=0)
        val_rec = recall_score(all_targets, all_preds, zero_division=0)
        val_acc = (all_preds == all_targets).mean()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_prec'].append(val_prec)
        history['val_rec'].append(val_rec)

        # Print progress
        elapsed = time.time() - t_start
        eta = (elapsed / epoch) * (args.epochs - epoch)

        print(f"  Epoch {epoch:>3}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f} "
              f"P: {val_prec:.4f} R: {val_rec:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"ETA: {eta/60:.0f}min")

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_prec': val_prec,
                'val_rec': val_rec,
                'history': history,
                'config': {
                    'input_channels': config.INPUT_CHANNELS,
                    'num_points': args.num_points,
                    'dropout': config.DROPOUT,
                    'voxel_size': args.voxel,
                    'training_fragments': frag_names,
                },
            }, checkpoint_path)
            print(f"    ** Saved best model (F1={val_f1:.4f}) **")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # ---- Summary ----
    total_time = time.time() - t_start

    print(f"\n{'#'*70}")
    print("TRAINING COMPLETE")
    print(f"{'#'*70}")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"  Trained on: {frag_names}")

    # Save history
    history_path = os.path.join(config.RESULTS_DIR, "training_history.npz")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    np.savez(history_path, **{k: np.array(v) for k, v in history.items()})
    print(f"  History: {history_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PointNet++")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--num_points", type=int, default=config.NUM_POINTS_PER_SAMPLE)
    parser.add_argument("--voxel", type=float, default=config.VOXEL_SIZE)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--samples_per_frag", type=int, default=50000)
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--gpu", type=str, default=config.DEVICE)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()