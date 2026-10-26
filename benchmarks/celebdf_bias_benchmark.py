"""Cross-dataset replication of resolution, smoothing, and center-prior bias."""
import argparse, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from torchvision import transforms
sys.path.insert(0,os.path.dirname(__file__))
from model_cnn import CNNBaseline
from controlled_benchmark import normalize_map, perturbation_metrics, resample, gaussian
from prior_attack_benchmark import priors

cv2.setNumThreads(4); torch.set_num_threads(4)
TFM=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def collect(root,limit,frames):
    ds=[os.path.join(root,d) for d in sorted(os.listdir(root))]; return [d for d in ds if os.path.isdir(d) and all(os.path.exists(os.path.join(d,f"frame_{i}.jpg")) for i in range(frames))][:limit]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="/data/CelebDF_8Frames/Celeb-synthesis")
    p.add_argument("--checkpoint",default="./checkpoints/resnet50_best.pt"); p.add_argument("--videos",type=int,default=80)
    p.add_argument("--frames",type=int,default=8); p.add_argument("--steps",type=int,default=10)
    p.add_argument("--output",default="./results/celebdf_bias_benchmark.txt"); args=p.parse_args(); device=torch.device("cuda")
    model=CNNBaseline(num_classes=2,pretrained=False).to(device); ck=torch.load(args.checkpoint,map_location=device); model.load_state_dict(ck.get("model",ck)); model.eval()
    acts=None; grads=None
    def fh(m,i,o):
        nonlocal acts; acts=o
    def bh(m,gi,go):
        nonlocal grads; grads=go[0]
    model.backbone.layer4.register_forward_hook(fh); model.backbone.layer4.register_full_backward_hook(bh)
    methods=["vanilla","gauss4","vanilla7","center_gaussian","face_ellipse","random","gradcam"]; rows={m:[] for m in methods}; seen=skipped=0; sid=0
    ds=collect(args.data_root,args.videos,args.frames); print(f"Celeb-synthesis: {len(ds)} videos",flush=True)
    for vi,path in enumerate(ds):
        for t in range(args.frames):
            rgb=cv2.cvtColor(cv2.imread(os.path.join(path,f"frame_{t}.jpg")),cv2.COLOR_BGR2RGB); frame=TFM(rgb).unsqueeze(0).to(device).requires_grad_(True)
            logits=model.forward_frame(frame); seen+=1
            if F.softmax(logits,1)[0,1].item()<.5: skipped+=1; continue
            model.zero_grad(); logits[0,1].backward(); vanilla=normalize_map(frame.grad.abs().mean(1,keepdim=True)).detach()
            gc=normalize_map(F.interpolate(F.relu((grads.detach().mean((2,3),keepdim=True)*acts.detach()).sum(1,keepdim=True)),(224,224),mode="bilinear",align_corners=False))
            cams=priors(device,42+sid); cams.update({"vanilla":vanilla,"gauss4":gaussian(vanilla,4),"vanilla7":resample(vanilla,7),"gradcam":gc})
            for name,cam in cams.items(): rows[name].append(perturbation_metrics(model,frame.detach(),cam,args.steps))
            sid+=1
        if (vi+1)%10==0: print(f"  {vi+1}/{len(ds)}",flush=True)
    with open(args.output,"w",encoding="utf-8") as f:
        f.write("method,dataset,n,seen,skipped,coverage,ad_mean,ad_std,ai,insertion_auc,deletion_auc,orig_conf\n")
        for method in methods:
            a=np.asarray(rows[method]); f.write(f"{method},CelebDF-v2,{len(a)},{seen},{skipped},{len(a)/seen:.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
    print(f"wrote {args.output}",flush=True)
if __name__=="__main__": main()
