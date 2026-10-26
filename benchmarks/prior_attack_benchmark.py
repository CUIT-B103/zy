"""Test whether model-agnostic spatial priors can game faithfulness metrics."""
import argparse, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0,os.path.dirname(__file__))
from model_cnn import CNNBaseline
from controlled_benchmark import heldout, normalize_map, perturbation_metrics, resample

cv2.setNumThreads(4); torch.set_num_threads(4)
STYLES=["Deepfakes","Face2Face","FaceShifter","DeepFakeDetection"]
TFM=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def priors(device,seed):
    y,x=torch.meshgrid(torch.linspace(-1,1,224,device=device),torch.linspace(-1,1,224,device=device),indexing="ij")
    center=torch.exp(-(x*x/(2*.32**2)+y*y/(2*.42**2))).view(1,1,224,224)
    ellipse=((x/.58).pow(2)+(y/.78).pow(2)<=1).float().view(1,1,224,224)
    gen=torch.Generator(device=device).manual_seed(seed)
    random=torch.rand((1,1,224,224),generator=gen,device=device)
    return {"center_gaussian":center,"face_ellipse":ellipse,"random":random}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/FFPlusPlus_Extracted_8Frames")
    p.add_argument("--checkpoint",default="./checkpoints/resnet50_best.pt")
    p.add_argument("--frames",type=int,default=8); p.add_argument("--max-videos",type=int,default=40); p.add_argument("--steps",type=int,default=10)
    p.add_argument("--output",default="./results/prior_attack_benchmark.txt"); args=p.parse_args()
    device=torch.device("cuda"); model=CNNBaseline(num_classes=2,pretrained=False).to(device); ck=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ck.get("model",ck)); model.eval(); methods=["vanilla","vanilla7","permuted7","center_gaussian","face_ellipse","random"]
    rows={s:{m:[] for m in methods} for s in STYLES}; seen={s:0 for s in STYLES}; skipped={s:0 for s in STYLES}; sample_id=0
    for style in STYLES:
        ds=heldout(args.data_root,style,args.frames,args.max_videos); print(f"{style}: {len(ds)} videos",flush=True)
        for vi,path in enumerate(ds):
            for t in range(args.frames):
                rgb=cv2.cvtColor(cv2.imread(os.path.join(path,f"frame_{t}.jpg")),cv2.COLOR_BGR2RGB)
                frame=TFM(rgb).unsqueeze(0).to(device).requires_grad_(True); logits=model.forward_frame(frame); seen[style]+=1
                if F.softmax(logits,1)[0,1].item()<.5: skipped[style]+=1; continue
                model.zero_grad(); logits[0,1].backward(); vanilla=normalize_map(frame.grad.abs().mean(1,keepdim=True)).detach(); vanilla7=resample(vanilla,7)
                gen=torch.Generator(device=device).manual_seed(100000+sample_id); perm=torch.randperm(49,generator=gen,device=device)
                low=F.interpolate(vanilla,(7,7),mode="area").flatten()[perm].view(1,1,7,7)
                permuted=normalize_map(F.interpolate(low,(224,224),mode="bilinear",align_corners=False)); cams=priors(device,42+sample_id)
                cams.update({"vanilla":vanilla,"vanilla7":vanilla7,"permuted7":permuted})
                for name,cam in cams.items(): rows[style][name].append(perturbation_metrics(model,frame.detach(),cam,args.steps))
                sample_id+=1
            if (vi+1)%5==0: print(f"  {vi+1}/{len(ds)}",flush=True)
    with open(args.output,"w",encoding="utf-8") as f:
        f.write("method,style,n,seen,skipped,coverage,ad_mean,ad_std,ai,insertion_auc,deletion_auc,orig_conf\n"); totals={m:[] for m in methods}
        for style in STYLES:
            for method in methods:
                a=np.asarray(rows[style][method]); totals[method].extend(rows[style][method]); f.write(f"{method},{style},{len(a)},{seen[style]},{skipped[style]},{len(a)/seen[style]:.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
        ts=sum(seen.values()); sk=sum(skipped.values())
        for method,data in totals.items():
            a=np.asarray(data); f.write(f"{method},OVERALL,{len(a)},{ts},{sk},{len(a)/ts:.6f},{a[:,0].mean():.4f},{a[:,0].std(ddof=1):.4f},{100*a[:,1].mean():.4f},{a[:,2].mean():.6f},{a[:,3].mean():.6f},{a[:,4].mean():.6f}\n")
    print(f"wrote {args.output}",flush=True)
if __name__=="__main__": main()
