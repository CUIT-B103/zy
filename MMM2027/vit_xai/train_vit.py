"""
ViT-B/16 6-Style balanced detector training.
数据流与 train.py 完全一致，只换 Backbone
"""
import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_vit import ViTBaseline


# ============ 数据加载（从 train.py 直接搬运） ============

class FFPlusPlusBalancedImageDataset(Dataset):
    def __init__(self, data_root, num_frames=8, is_train=True, split_ratio=0.8, samples_per_style=500, seed=42):
        self.data_root = data_root
        self.num_frames = num_frames
        self.is_train = is_train
        self.samples_per_style = samples_per_style
        self.split_ratio = split_ratio

        self.fake_styles = [
            {"name": "Deepfakes", "folder": "Deepfakes"},
            {"name": "Face2Face", "folder": "Face2Face"},
            {"name": "FaceSwap", "folder": "FaceSwap"},
            {"name": "NeuralTextures", "folder": "NeuralTextures"},
            {"name": "FaceShifter", "folder": "FaceShifter"},
            {"name": "DeepFakeDetection", "folder": "DeepFakeDetection"},
        ]
        self.real_folder = "original"

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        np.random.seed(seed)
        self.final_samples = self._build_leakage_free_dataset()

        print(f"  [{'TRAIN' if is_train else 'VAL'}] 样本数: {len(self.final_samples)}")
        print(f"     Real: {sum(1 for _, l in self.final_samples if l == 0)}, Fake: {sum(1 for _, l in self.final_samples if l == 1)}")

    def _collect_video_dirs(self, folder_name):
        target_dir = os.path.join(self.data_root, folder_name)
        video_dirs = []
        if os.path.exists(target_dir):
            for sub_dir in os.listdir(target_dir):
                full_sub_dir = os.path.join(target_dir, sub_dir)
                if os.path.isdir(full_sub_dir):
                    all_ok = all(
                        os.path.exists(os.path.join(full_sub_dir, f"frame_{i}.jpg"))
                        for i in range(self.num_frames)
                    )
                    if all_ok:
                        video_dirs.append(full_sub_dir)
        return sorted(video_dirs)

    def _build_leakage_free_dataset(self):
        samples = []
        fake_train_by_style = {}
        fake_val_by_style = {}

        for style in self.fake_styles:
            v_dirs = self._collect_video_dirs(style["folder"])
            if len(v_dirs) > 0:
                split_idx = int(len(v_dirs) * self.split_ratio)
                fake_train_by_style[style["name"]] = v_dirs[:split_idx]
                fake_val_by_style[style["name"]] = v_dirs[split_idx:]

        real_v_dirs = self._collect_video_dirs(self.real_folder)
        real_split_idx = int(len(real_v_dirs) * self.split_ratio)
        real_train_pool = real_v_dirs[:real_split_idx]
        real_val_pool = real_v_dirs[real_split_idx:]

        if self.is_train:
            total_fake_train = 0
            for style_name, train_pool in fake_train_by_style.items():
                if len(train_pool) == 0: continue
                selected = np.random.choice(train_pool, self.samples_per_style, replace=(len(train_pool) < self.samples_per_style))
                for d in selected: samples.append((d, 1))
                total_fake_train += len(selected)
            if len(real_train_pool) > 0:
                selected_real = np.random.choice(real_train_pool, total_fake_train, replace=(len(real_train_pool) < total_fake_train))
                for d in selected_real: samples.append((d, 0))
        else:
            val_per_style = min(100, self.samples_per_style // 5)
            total_fake_val = 0
            for style_name, val_pool in fake_val_by_style.items():
                if len(val_pool) == 0: continue
                selected_val = val_pool[:val_per_style]
                for d in selected_val: samples.append((d, 1))
                total_fake_val += len(selected_val)
            if len(real_val_pool) > 0:
                selected_real_val = np.random.choice(real_val_pool, total_fake_val, replace=(len(real_val_pool) < total_fake_val))
                for d in selected_real_val: samples.append((d, 0))

        np.random.shuffle(samples)
        return samples

    def __len__(self):
        return len(self.final_samples)

    def __getitem__(self, idx):
        import cv2
        video_dir, label = self.final_samples[idx]
        sampled_tensors = []
        for i in range(self.num_frames):
            img_path = os.path.join(video_dir, f"frame_{i}.jpg")
            img = cv2.imread(img_path)
            if img is None:
                frame_tensor = torch.zeros((3, 224, 224))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(img)
            sampled_tensors.append(frame_tensor)
        video_tensor = torch.stack(sampled_tensors, dim=0)
        return video_tensor, label


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lr[i] if isinstance(lr, list) else lr

    def _get_lr(self):
        if self.current_step <= self.warmup_steps:
            alpha = self.current_step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_decay for base_lr in self.base_lrs]

    def get_last_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]


def parse_args():
    parser = argparse.ArgumentParser(description="ViT-B/16 FF++ Training")
    parser.add_argument("--data-root", type=str, default="data/FFPlusPlus_Extracted_8Frames")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-style", type=int, default=500)
    parser.add_argument("--cpu-cores", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, device, grad_clip, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (videos, labels) in enumerate(dataloader):
        B, T, C, H, W = videos.shape
        if videos.sum() == 0: continue

        frames = videos.view(B * T, C, H, W).to(device)
        labels_per_frame = labels.unsqueeze(1).expand(B, T).reshape(B * T).to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model.forward_frame(frames)
            loss = criterion(logits, labels_per_frame)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels_per_frame).sum().item()
        total += labels_per_frame.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f} | Acc: {correct/total:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(dataloader), correct / total


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    for videos, labels in dataloader:
        if videos.sum() == 0: continue
        videos, labels = videos.to(device), labels.to(device)
        logits_video = model(videos)
        loss = criterion(logits_video, labels)
        total_loss += loss.item()

        probs = F.softmax(logits_video, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    all_labels, all_probs = np.array(all_labels), np.array(all_probs)
    val_auc = 0.5 if len(np.unique(all_labels)) < 2 else roc_auc_score(all_labels, all_probs)
    val_acc = accuracy_score(all_labels, (all_probs >= 0.5).astype(int))
    return total_loss / len(dataloader), val_acc, val_auc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] ViT-B/16 | device={device} | workers={args.num_workers}")

    train_dataset = FFPlusPlusBalancedImageDataset(
        data_root=args.data_root, num_frames=args.num_frames,
        is_train=True, samples_per_style=args.samples_per_style, seed=args.seed
    )
    val_dataset = FFPlusPlusBalancedImageDataset(
        data_root=args.data_root, num_frames=args.num_frames,
        is_train=False, samples_per_style=args.samples_per_style, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = ViTBaseline(num_classes=2, pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=len(train_loader) * args.warmup_epochs,
        total_steps=len(train_loader) * args.epochs,
        min_lr=args.min_lr
    )
    scaler = GradScaler()

    start_epoch, best_auc = 0, 0.0
    if args.resume:
        print(f"[恢复] {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt.get("best_auc", 0.0)

    print(f"\n[ViT-B/16 training] epochs={args.epochs}, batch={args.batch_size}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler, device, args.grad_clip, epoch
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        print(f"\n[Epoch {epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
              f"Time: {time.time() - t0:.1f}s")

        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            print(f"  >>> 最佳 AUC: {best_auc:.4f}")

        checkpoint = {
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "best_auc": best_auc, "val_acc": val_acc
        }
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        if is_best:
            torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))

    print(f"\n[完成] 最佳 AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
