"""Resolution- and smoothing-controlled faithfulness benchmark."""
import argparse, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from model_cnn import CNNBaseline

cv2.setNumThreads(4); torch.set_num_threads(4); torch.set_num_interop_threads(1)
STYLES = ["Deepfakes", "Face2Face", "FaceShifter", "DeepFakeDetection"]
TFM = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                          transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
MEAN = [.485,.456,.406]; STD = [.229,.224,.225]

def heldout(root, style, frames, limit):
    ds=[os.path.join(root,style,d) for d in sorted(os.listdir(os.path.join(root,style)))]
    ds=[d for d in ds if os.path.isdir(d) and all(os.path.exists(os.path.join(d,f"frame_{i}.jpg")) for i in range(frames))]
    return ds[int(.8*len(ds)):][:limit]

def normalize_map(x):
    lo=x.amin((2,3),keepdim=True); hi=x.amax((2,3),keepdim=True)
    return (x-lo)/(hi-lo+1e-8)

def gaussian(cam, sigma):
    arr=cam[0,0].cpu().numpy(); arr=cv2.GaussianBlur(arr,(0,0),sigmaX=sigma,sigmaY=sigma)
    return normalize_map(torch.from_numpy(arr).to(cam.device).view(1,1,*arr.shape))

def resample(cam, size):
    low=F.interpolate(cam,(size,size),mode="area")
    return normalize_map(F.interpolate(low,(224,224),mode="bilinear",align_corners=False))

def confidence(model,x): return F.softmax(model.forward_frame(x),1)[:,1]

@torch.no_grad()
def perturbation_metrics(model, frame, saliency, steps=10):
    mean=torch.tensor(MEAN,device=frame.device).view(1,3,1,1); std=torch.tensor(STD,device=frame.device).view(1,3,1,1)
    raw=(frame*std+mean).clamp(0,1); baseline=torch.full_like(raw,.5)
    flat=saliency.flatten(); order=torch.argsort(flat,descending=True); n=flat.numel()
    p0=confidence(model,frame).item(); curves_i=[]; curves_d=[]
    for step in range(steps+1):
        k=round(n*step/steps); mask=torch.zeros(n,device=frame.device)
        if k: mask[order[:k]]=1
        mask=mask.view(1,1,224,224)
        inserted=baseline*(1-mask)+raw*mask; deleted=raw*(1-mask)+baseline*mask
        curves_i.append(confidence(model,(inserted-mean)/std).item())
        curves_d.append(confidence(model,(deleted-mean)/std).item())
    keep=torch.zeros(n,device=frame.device); keep[order[:max(1,n//10)]]=1; keep=keep.view(1,1,224,224)
    pk=confidence(model,(raw*keep-mean)/std).item()
    ad=max(0,(p0-pk)/(p0+1e-8))*100; ai=float(pk>=p0)
    x=np.linspace(0,1,steps+1)
    return ad,ai,float(np.trapz(curves_i,x)),float(np.trapz(curves_d,x)),p0

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/FFPlusPlus_Extracted_8Frames")
    p.add_argument("--checkpoint",default="./checkpoints/resnet50_best.pt")
    p.add_argument("--frames",type=int,default=8); p.add_argument("--max-videos",type=int,default=40)
    p.add_argument("--steps",type=int,default=10); p.add_argument("--output",default="./results/controlled_benchmark.txt")
    args=p.parse_args(); device=torch.device("cuda")
    model=CNNBaseline(num_classes=2,pretrained=False).to(device); ck=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ck.get("model",ck)); model.eval(); acts=None; grads=None
    def fh(m,i,o):
        nonlocal acts; acts=o
    def bh(m,gi,go):
        nonlocal grads; grads=go[0]
    model.backbone.layer4.register_forward_hook(fh); model.backbone.layer4.register_full_backward_hook(bh)
    methods=["vanilla","gauss1","gauss2","gauss4","down112","down56","down28","down14","down7","gradcam"]
    rows={s:{m:[] for m in methods} for s in STYLES}; seen={s:0 for s in STYLES}; skipped={s:0 for s in STYLES}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    for style in STYLES:
        ds=heldout(args.data_root,style,args.frames,args.max_videos); print(f"{style}: {len(ds)} videos",flush=True)
        for vi,path in enumerate(ds):
            for t in range(args.frames):
                rgb=cv2.cvtColor(cv2.imread(os.path.join(path,f"frame_{t}.jpg")),cv2.COLOR_BGR2RGB)
                frame=TFM(rgb).unsqueeze(0).to(device).requires_grad_(True); logits=model.forward_frame(frame)
                seen[style]+=1
                if F.softmax(logits,1)[0,1].item() < .5:
                    skipped[style]+=1
                    continue
                model.zero_grad(); logits[0,1].backward(); vanilla=normalize_map(frame.grad.abs().mean(1,keepdim=True)).detach()
                weights=grads.detach().mean((2,3),keepdim=True); gc=F.relu((weights*acts.detach()).sum(1,keepdim=True))
                gc=normalize_map(F.interpolate(gc,(224,224),mode="bilinear",align_corners=False))
                cams={"vanilla":vanilla,"gauss1":gaussian(vanilla,1),"gauss2":gaussian(vanilla,2),
                      "gauss4":gaussian(vanilla,4),"down112":resample(vanilla,112),"down56":resample(vanilla,56),
                      "down28":resample(vanilla,28),"down14":resample(vanilla,14),"down7":resample(vanilla,7),"gradcam":gc}
                for name,cam in cams.items(): rows[style][name].append(perturbation_metrics(model,frame.detach(),cam,args.steps))
            if (vi+1)%5==0: print(f"  {vi+1}/{len(ds)}",flush=True)
    with open(args.output,"w",encoding="utf-8") as f:
        f.write("method,style,n,seen,skipped,coverage,ad_mean,ad_std,ai,insertion_auc,deletion_auc,orig_conf\n")
        totals={m:[] for m in methods}
        for style in STYLES:
            for method in methods:
                a=np.asarray(rows[style][method]); totals[method].extend(rows[style][method])
                coverage=len(a)/max(seen[style],1)
                f.write(f"{method},{style},{len(a)},{seen[style]},{skipped[style]},{coverage:.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
        for method,data in totals.items():
            a=np.asarray(data); total_seen=sum(seen.values()); total_skip=sum(skipped.values())
            f.write(f"{method},OVERALL,{len(a)},{total_seen},{total_skip},{len(a)/max(total_seen,1):.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
    print(f"wrote {args.output}",flush=True)
if __name__=="__main__": main()
