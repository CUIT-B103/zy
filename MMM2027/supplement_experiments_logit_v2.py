import argparse, csv, json, hashlib, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from torchvision import transforms
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "benchmarks"))
from model_cnn import CNNBaseline
from controlled_benchmark import perturbation_metrics

STYLES = ["Deepfakes", "Face2Face", "FaceShifter", "DeepFakeDetection"]
MEAN, STD = [.485, .456, .406], [.229, .224, .225]
TFM = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def normalize(x):
    lo, hi = x.amin((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)

def list_videos(root, style, frames, limit):
    base = os.path.join(root, style)
    ds = [os.path.join(base, d) for d in sorted(os.listdir(base)) if os.path.isdir(os.path.join(base, d))]
    ds = [d for d in ds if all(os.path.exists(os.path.join(d, f"frame_{i}.jpg")) for i in range(frames))]
    return ds[int(.8 * len(ds)):][:limit]

def resample(sal, size):
    return normalize(F.interpolate(F.interpolate(sal, (size, size), mode="area"), (224, 224), mode="bilinear", align_corners=False))

def gaussian(sal, sigma):
    arr = sal[0, 0].detach().cpu().numpy()
    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return normalize(torch.from_numpy(arr).to(sal.device).view(1, 1, 224, 224))

def geometry(sal):
    a = sal[0, 0].detach().cpu().numpy()
    order = torch.argsort(sal[0, 0].flatten(), descending=True).cpu().numpy()
    mask = np.zeros(a.size, np.uint8); mask[order[:max(1, a.size // 10)]] = 1; mask = mask.reshape(a.shape)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
    tv = np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean()
    return float(areas.max() / mask.sum()) if areas.size else 0., int(len(areas)), float(tv)

def score(model, x, label):
    return F.softmax(model.forward_frame(x), 1)[:, label]

@torch.no_grad()
def metrics(model, frame, sal, steps, label):
    mean = torch.tensor(MEAN, device=frame.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=frame.device).view(1, 3, 1, 1)
    raw, base = (frame * std + mean).clamp(0, 1), torch.full_like(frame, .5)
    flat, order = sal[0, 0].flatten(), torch.argsort(sal[0, 0].flatten(), descending=True)
    n, p0, ins, dele = flat.numel(), score(model, frame, label).item(), [], []
    for j in range(steps + 1):
        k = round(n * j / steps); m = torch.zeros(n, device=frame.device); m[order[:k]] = 1; m = m.view(1, 1, 224, 224)
        ins.append(score(model, (base * (1-m) + raw*m - mean) / std, label).item())
        dele.append(score(model, (raw * (1-m) + base*m - mean) / std, label).item())
    m = torch.zeros(n, device=frame.device); m[order[:max(1, n // 10)]] = 1; m = m.view(1, 1, 224, 224)
    pk = score(model, (raw*m - mean) / std, label).item()
    x = np.linspace(0, 1, steps + 1)
    return [max(0, (p0-pk)/(p0+1e-8))*100, 100*float(pk >= p0), float(np.trapz(ins, x)), float(np.trapz(dele, x)), *geometry(sal)]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--data-root", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--output", required=True)
    p.add_argument("--frames", type=int, default=8); p.add_argument("--max-videos", type=int, default=20); p.add_argument("--noise-samples", type=int, default=30); p.add_argument("--ig-steps", type=int, default=32); p.add_argument("--insertion-steps", type=int, default=5); p.add_argument("--real", action="store_true")
    a = p.parse_args(); cv2.setNumThreads(4); torch.set_num_threads(4)
    torch.manual_seed(42); np.random.seed(42); dev = torch.device("cuda")
    model = CNNBaseline(num_classes=2, pretrained=False).to(dev); ck = torch.load(a.checkpoint, map_location=dev); model.load_state_dict(ck.get("model", ck)); model.eval(); rows = []
    label, styles = (0, ["original"]) if a.real else (1, STYLES)
    coverage = {}
    for style in styles:
        coverage[style] = dict(seen=0, accepted=0)
        for vi, path in enumerate(list_videos(a.data_root, style, a.frames, a.max_videos)):
            for t in range(a.frames):
                rgb = cv2.cvtColor(cv2.imread(os.path.join(path, f"frame_{t}.jpg")), cv2.COLOR_BGR2RGB); x = TFM(rgb).unsqueeze(0).to(dev).requires_grad_(True)
                logits = model.forward_frame(x); prob = F.softmax(logits, 1)[0, label].item()
                coverage[style]['seen'] += 1
                if logits.argmax(1).item() != label: continue
                coverage[style]['accepted'] += 1
                model.zero_grad(); logits[0, label].backward(); vanilla = normalize(x.grad.abs().mean(1, keepdim=True)).detach()
                if label == 1:
                    reference = perturbation_metrics(model, x.detach(), vanilla, a.insertion_steps)
                    current = metrics(model, x.detach(), vanilla, a.insertion_steps, label)
                    np.testing.assert_allclose(current[:4], [reference[0], reference[1]*100, *reference[2:4]], atol=1e-6, rtol=0)
                gs = []
                for _ in range(a.noise_samples):
                    z = (x.detach() + torch.randn_like(x) * .1).requires_grad_(True); model.zero_grad(); model.forward_frame(z)[0, label].backward(); gs.append(z.grad.abs().mean(1, keepdim=True).detach())
                smooth = normalize(torch.stack(gs).mean(0)); base = (-torch.tensor(MEAN, device=dev).view(1,3,1,1) / torch.tensor(STD, device=dev).view(1,3,1,1)).expand_as(x); total = torch.zeros_like(x)
                for s in range(1, a.ig_steps + 1):
                    z = (base + (x.detach()-base) * s / a.ig_steps).requires_grad_(True); model.zero_grad(); model.forward_frame(z)[0, label].backward(); total += z.grad.detach()
                ig = normalize(((x.detach()-base) * total / a.ig_steps).abs().mean(1, keepdim=True))
                maps = [("Vanilla", vanilla), ("Vanilla-7", resample(vanilla,7)), ("SmoothGrad", smooth), ("SmoothGrad-7", resample(smooth,7)), ("IntegratedGradients", ig), ("IntegratedGradients-7", resample(ig,7))]
                maps.append(("Gaussian-sigma4", gaussian(vanilla, 4)))
                for name, sal in maps:
                    rows.append([style, os.path.basename(path), t, label, name, prob] + metrics(model, x.detach(), sal, a.insertion_steps, label))
            if (vi + 1) % 5 == 0: print(style, vi + 1, flush=True)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    fields = ['style','video_id','frame_idx','label','method','orig_conf','AD@10','AI@10','insertion','deletion','max_cc_frac','n_components','TV','noise_sigma_normalized','noise_samples','ig_steps','ig_baseline','insertion_steps','deletion_steps','channel_reduce','seed']
    rules = {'Vanilla':'abs_gradient_then_channel_mean','Vanilla-7':'abs_gradient_then_channel_mean_after_7x7','SmoothGrad':'mean_abs_gradient_over_noise_channel_mean','SmoothGrad-7':'mean_abs_gradient_over_noise_then_7x7','IntegratedGradients':'abs_path_integrated_attribution_then_channel_mean','IntegratedGradients-7':'abs_path_integrated_attribution_then_channel_mean_then_7x7','Gaussian-sigma4':'spatial_gaussian_on_vanilla_sigma4_pixels'}
    with open(a.output, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(fields)
        for r in rows: w.writerow(r + [.1, a.noise_samples, a.ig_steps, 'black', a.insertion_steps, a.insertion_steps, rules[r[4]], 42])
    print('wrote', a.output, 'rows', len(rows), flush=True)
    with open(a.checkpoint, 'rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    metadata = dict(vars(a), coverage=coverage, attribution_target='class_logit', metric_target='class_probability', ad_fill='black_pixel_0', curve_fill='gray_pixel_0.5', normalization='per_map_minmax', tie_rule='torch.argsort_descending_default_same_as_main', geometry_connectivity=8, gaussian_sigma_pixels=4, checkpoint_sha256=digest, seed=42, torch_version=torch.__version__)
    with open(a.output + '.json', 'w') as f: json.dump(metadata, f, indent=2)
if __name__ == '__main__': main()
