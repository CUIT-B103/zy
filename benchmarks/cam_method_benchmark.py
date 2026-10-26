"""Fair comparison of five CAM families under one perturbation protocol."""
import argparse, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from model_cnn import CNNBaseline
from controlled_benchmark import heldout, normalize_map, resample, perturbation_metrics

cv2.setNumThreads(4); torch.set_num_threads(4)
STYLES=["Deepfakes","Face2Face","FaceShifter","DeepFakeDetection"]
TFM=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def up(cam): return normalize_map(F.interpolate(cam,(224,224),mode="bilinear",align_corners=False))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/FFPlusPlus_Extracted_8Frames")
    p.add_argument("--checkpoint",default="./checkpoints/resnet50_best.pt")
    p.add_argument("--frames",type=int,default=8); p.add_argument("--max-videos",type=int,default=20)
    p.add_argument("--steps",type=int,default=5); p.add_argument("--score-topk",type=int,default=32)
    p.add_argument("--output",default="./results/cam_method_benchmark.txt")
    args=p.parse_args(); device=torch.device("cuda")
    model=CNNBaseline(num_classes=2,pretrained=False).to(device); ck=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ck.get("model",ck)); model.eval(); acts=None; grads=None
    def fh(m,i,o):
        nonlocal acts; acts=o
    def bh(m,gi,go):
        nonlocal grads; grads=go[0]
    model.backbone.layer4.register_forward_hook(fh); model.backbone.layer4.register_full_backward_hook(bh)
    methods=["vanilla_native","vanilla_7","gradcam","gradcampp","groupcam","fastscorecam"]
    rows={s:{m:[] for m in methods} for s in STYLES}; seen={s:0 for s in STYLES}; skipped={s:0 for s in STYLES}
    for style in STYLES:
        ds=heldout(args.data_root,style,args.frames,args.max_videos); print(f"{style}: {len(ds)} videos",flush=True)
        for vi,path in enumerate(ds):
            for t in range(args.frames):
                rgb=cv2.cvtColor(cv2.imread(os.path.join(path,f"frame_{t}.jpg")),cv2.COLOR_BGR2RGB)
                frame=TFM(rgb).unsqueeze(0).to(device).requires_grad_(True); logits=model.forward_frame(frame); seen[style]+=1
                if F.softmax(logits,1)[0,1].item()<.5: skipped[style]+=1; continue
                model.zero_grad(); logits[0,1].backward(); a=acts.detach().clone(); g=grads.detach().clone()
                vanilla=normalize_map(frame.grad.abs().mean(1,keepdim=True)).detach()
                gc=up(F.relu((g.mean((2,3),keepdim=True)*a).sum(1,keepdim=True)))
                g2=g.pow(2); g3=g2*g; denom=2*g2+(a*g3).sum((2,3),keepdim=True)
                alpha=g2/(denom+1e-8); wpp=(alpha*F.relu(g)).sum((2,3),keepdim=True)
                gcpp=up(F.relu((wpp*a).sum(1,keepdim=True)))
                c=a.shape[1]; groups=32; gs=c//groups; group_maps=a.view(1,groups,gs,7,7).sum(2)
                group_scores=[]
                with torch.no_grad():
                    for gi in range(groups):
                        masked=torch.zeros_like(a); masked[:,gi*gs:(gi+1)*gs]=a[:,gi*gs:(gi+1)*gs]
                        score=F.softmax(model.backbone.fc(model.backbone.avgpool(masked).flatten(1)),1)[0,1]
                        group_scores.append(score)
                group_w=torch.stack(group_scores).view(1,groups,1,1)
                groupcam=up(F.relu((group_w*group_maps).sum(1,keepdim=True)))
                mags=a[0].flatten(1).mean(1); idx=torch.topk(mags,min(args.score_topk,c)).indices
                selected=a[:,idx]; amin=selected.amin((2,3),keepdim=True); amax=selected.amax((2,3),keepdim=True)
                masks=F.interpolate((selected-amin)/(amax-amin+1e-8),(224,224),mode="bilinear",align_corners=False)
                with torch.no_grad(): scores=F.softmax(model.forward_frame(frame.detach()*masks.transpose(0,1)),1)[:,1]
                scorecam=up(F.relu((scores.view(1,-1,1,1)*selected).sum(1,keepdim=True)))
                cams={"vanilla_native":vanilla,"vanilla_7":resample(vanilla,7),"gradcam":gc,
                      "gradcampp":gcpp,"groupcam":groupcam,"fastscorecam":scorecam}
                for name,cam in cams.items(): rows[style][name].append(perturbation_metrics(model,frame.detach(),cam,args.steps))
            if (vi+1)%5==0: print(f"  {vi+1}/{len(ds)}",flush=True)
    with open(args.output,"w",encoding="utf-8") as f:
        f.write("method,style,n,seen,skipped,coverage,ad_mean,ad_std,ai,insertion_auc,deletion_auc,orig_conf\n")
        totals={m:[] for m in methods}
        for style in STYLES:
            for method in methods:
                a=np.asarray(rows[style][method]); totals[method].extend(rows[style][method])
                if len(a)==0:
                    f.write(f"{method},{style},0,{seen[style]},{skipped[style]},0.000000,nan,nan,nan,nan,nan,nan\n")
                    continue
                f.write(f"{method},{style},{len(a)},{seen[style]},{skipped[style]},{len(a)/max(seen[style],1):.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
        total_seen=sum(seen.values()); total_skip=sum(skipped.values())
        for method,data in totals.items():
            a=np.asarray(data); f.write(f"{method},OVERALL,{len(a)},{total_seen},{total_skip},{len(a)/total_seen:.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
    print(f"wrote {args.output}",flush=True)
if __name__=="__main__": main()
