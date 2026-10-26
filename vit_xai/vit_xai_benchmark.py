"""Cross-backbone faithfulness controls for a ViT-B/16 detector."""
import argparse
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from model_vit import ViTBaseline


cv2.setNumThreads(4)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

STYLES = ["Deepfakes", "Face2Face", "FaceShifter", "DeepFakeDetection"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TFM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def heldout(root, style, frames, limit):
    base = os.path.join(root, style)
    videos = [os.path.join(base, name) for name in sorted(os.listdir(base))]
    videos = [path for path in videos if os.path.isdir(path) and all(
        os.path.exists(os.path.join(path, f"frame_{i}.jpg")) for i in range(frames)
    )]
    return videos[int(0.8 * len(videos)):][:limit]


def normalize_map(value):
    low = value.amin((2, 3), keepdim=True)
    high = value.amax((2, 3), keepdim=True)
    return (value - low) / (high - low + 1e-8)


def resample(value, size):
    low = F.interpolate(value, (size, size), mode="area")
    return normalize_map(F.interpolate(low, (224, 224), mode="bilinear", align_corners=False))


def center_gaussian(device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 224, device=device),
        torch.linspace(-1, 1, 224, device=device),
        indexing="ij",
    )
    value = torch.exp(-0.5 * ((xx / 0.42) ** 2 + (yy / 0.52) ** 2))
    return normalize_map(value.view(1, 1, 224, 224))


def confidence(model, value):
    return F.softmax(model.forward_frame(value), dim=1)[:, 1]


@torch.no_grad()
def metrics(model, frame, saliency, steps):
    mean = torch.tensor(MEAN, device=frame.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=frame.device).view(1, 3, 1, 1)
    raw = (frame * std + mean).clamp(0, 1)
    baseline = torch.full_like(raw, 0.5)
    order = torch.argsort(saliency.flatten(), descending=True)
    count = order.numel()
    original = confidence(model, frame).item()
    insertion, deletion = [], []
    for step in range(steps + 1):
        kept = round(count * step / steps)
        mask = torch.zeros(count, device=frame.device)
        if kept:
            mask[order[:kept]] = 1
        mask = mask.view(1, 1, 224, 224)
        inserted = baseline * (1 - mask) + raw * mask
        deleted = raw * (1 - mask) + baseline * mask
        insertion.append(confidence(model, (inserted - mean) / std).item())
        deletion.append(confidence(model, (deleted - mean) / std).item())
    kept = count // 10
    mask = torch.zeros(count, device=frame.device)
    mask[order[:kept]] = 1
    mask = mask.view(1, 1, 224, 224)
    retained = confidence(model, (raw * mask - mean) / std).item()
    ad = max(0, (original - retained) / (original + 1e-8)) * 100
    ai = float(retained >= original)
    axis = np.linspace(0, 1, steps + 1)
    return ad, ai, float(np.trapz(insertion, axis)), float(np.trapz(deletion, axis)), original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/FFPlusPlus_Extracted_8Frames")
    parser.add_argument("--checkpoint", default="./checkpoints/best.pt")
    parser.add_argument("--output", default="./results/vit_xai_benchmark.txt")
    parser.add_argument("--detail", default="./results/vit_xai_framewise.csv")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-videos", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = ViTBaseline(num_classes=2, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    activations = None
    gradients = None

    def forward_hook(module, inputs, output):
        nonlocal activations
        activations = output

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    target = model.backbone.encoder.layers[-1]
    target.register_forward_hook(forward_hook)
    target.register_full_backward_hook(backward_hook)

    methods = ["vanilla", "vanilla_14", "center_gaussian", "vit_gradcam"]
    rows = {style: {method: [] for method in methods} for style in STYLES}
    seen = {style: 0 for style in STYLES}
    skipped = {style: 0 for style in STYLES}
    detail_rows = []
    fixed_center = center_gaussian(device)

    for style in STYLES:
        videos = heldout(args.data_root, style, args.frames, args.max_videos)
        print(f"{style}: {len(videos)} videos", flush=True)
        for video_index, path in enumerate(videos):
            video_id = os.path.basename(path)
            for frame_index in range(args.frames):
                image = cv2.imread(os.path.join(path, f"frame_{frame_index}.jpg"))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame = TFM(image).unsqueeze(0).to(device).requires_grad_(True)
                logits = model.forward_frame(frame)
                probability = F.softmax(logits, dim=1)[0, 1].item()
                seen[style] += 1
                if probability < 0.5:
                    skipped[style] += 1
                    continue
                model.zero_grad(set_to_none=True)
                logits[0, 1].backward()
                vanilla = normalize_map(frame.grad.abs().mean(1, keepdim=True)).detach()
                patch_acts = activations.detach()[:, 1:, :]
                patch_grads = gradients.detach()[:, 1:, :]
                channel_weights = patch_grads.mean(dim=1, keepdim=True)
                token_map = F.relu((patch_acts * channel_weights).sum(dim=2))
                token_map = token_map.view(1, 1, 14, 14)
                vit_gradcam = normalize_map(F.interpolate(
                    token_map, (224, 224), mode="bilinear", align_corners=False
                ))
                maps = {
                    "vanilla": vanilla,
                    "vanilla_14": resample(vanilla, 14),
                    "center_gaussian": fixed_center,
                    "vit_gradcam": vit_gradcam,
                }
                for method, saliency in maps.items():
                    values = metrics(model, frame.detach(), saliency, args.steps)
                    rows[style][method].append(values)
                    detail_rows.append([
                        style, video_id, frame_index, method,
                        *[f"{value:.8f}" for value in values],
                    ])
            if (video_index + 1) % 5 == 0:
                print(f"  {video_index + 1}/{len(videos)}", flush=True)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.detail, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["style", "video", "frame", "method", "ad", "ai", "insertion", "deletion", "orig_conf"])
        writer.writerows(detail_rows)

    with open(args.output, "w", encoding="utf-8") as stream:
        stream.write("method,style,n,seen,skipped,coverage,ad_mean,ad_std,ai,insertion_auc,deletion_auc,orig_conf\n")
        totals = {method: [] for method in methods}
        for style in STYLES:
            for method in methods:
                values = np.asarray(rows[style][method])
                totals[method].extend(rows[style][method])
                stream.write(
                    f"{method},{style},{len(values)},{seen[style]},{skipped[style]},"
                    f"{len(values)/max(seen[style],1):.6f},{values[:,0].mean():.4f},"
                    f"{values[:,0].std(ddof=1):.4f},{100*values[:,1].mean():.4f},"
                    f"{values[:,2].mean():.6f},{values[:,3].mean():.6f},{values[:,4].mean():.6f}\n"
                )
        total_seen = sum(seen.values())
        total_skipped = sum(skipped.values())
        for method, data in totals.items():
            values = np.asarray(data)
            stream.write(
                f"{method},OVERALL,{len(values)},{total_seen},{total_skipped},"
                f"{len(values)/total_seen:.6f},{values[:,0].mean():.4f},"
                f"{values[:,0].std(ddof=1):.4f},{100*values[:,1].mean():.4f},"
                f"{values[:,2].mean():.6f},{values[:,3].mean():.6f},{values[:,4].mean():.6f}\n"
            )
    print(f"wrote {args.output} and {args.detail}", flush=True)


if __name__ == "__main__":
    main()
