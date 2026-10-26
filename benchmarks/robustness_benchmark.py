"""Faithfulness robustness across keep ratios and perturbation baselines."""
import argparse, csv, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from torchvision import transforms
sys.path.insert(0,os.path.dirname(__file__))
from model_cnn import CNNBaseline
from controlled_benchmark import heldout, normalize_map, resample
from prior_attack_benchmark import priors

cv2.setNumThreads(4); torch.set_num_threads(4)
STYLES=["Deepfakes","Face2Face","FaceShifter","DeepFakeDetection"]
TFM=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
MEAN=[.485,.456,.406]; STD=[.229,.224,.225]

def conf(model,x): return F.softmax(model.forward_frame(x),1)[:,1]

@torch.no_grad()
def evaluate(model,frame,saliency,baseline_name,ratios,steps):
    mean=torch.tensor(MEAN,device=frame.device).view(1,3,1,1); std=torch.tensor(STD,device=frame.device).view(1,3,1,1)
    raw=(frame*std+mean).clamp(0,1)
    if baseline_name=="black": baseline=torch.zeros_like(raw)
    elif baseline_name=="blur": baseline=F.avg_pool2d(F.pad(raw,(15,15,15,15),mode="reflect"),31,stride=1)
    else: baseline=torch.full_like(raw,.5)
    p0=conf(model,frame).item(); order=torch.argsort(saliency.flatten(),descending=True); n=order.numel(); out={}
    for ratio in ratios:
        k=max(1,round(n*ratio)); mask=torch.zeros(n,device=frame.device); mask[order[:k]]=1; mask=mask.view(1,1,224,224)
        kept=raw*mask+baseline*(1-mask); pk=conf(model,(kept-mean)/std).item()
        out[f"ad{int(ratio*100)}"]=max(0,(p0-pk)/(p0+1e-8))*100; out[f"ai{int(ratio*100)}"]=float(pk>=p0)
    ins=[]; dele=[]
    for step in range(steps+1):
        k=round(n*step/steps); mask=torch.zeros(n,device=frame.device)
        if k: mask[order[:k]]=1
        mask=mask.view(1,1,224,224); inserted=baseline*(1-mask)+raw*mask; deleted=raw*(1-mask)+baseline*mask
        ins.append(conf(model,(inserted-mean)/std).item()); dele.append(conf(model,(deleted-mean)/std).item())
    x=np.linspace(0,1,steps+1); out["insertion"]=float(np.trapz(ins,x)); out["deletion"]=float(np.trapz(dele,x)); out["p0"]=p0
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-root",default="data/FFPlusPlus_Extracted_8Frames")
    p.add_argument("--checkpoint",default="./checkpoints/resnet50_best.pt"); p.add_argument("--videos",type=int,default=20)
    p.add_argument("--frames",type=int,default=8); p.add_argument("--steps",type=int,default=5)
    p.add_argument("--detail",default="./results/robustness_framewise.csv"); p.add_argument("--output",default="./results/robustness_benchmark.txt")
    args=p.parse_args(); device=torch.device("cuda"); model=CNNBaseline(num_classes=2,pretrained=False).to(device); ck=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ck.get("model",ck)); model.eval(); acts=None; grads=None
    def fh(m,i,o):
        nonlocal acts; acts=o
    def bh(m,gi,go):
        nonlocal grads; grads=go[0]
    model.backbone.layer4.register_forward_hook(fh); model.backbone.layer4.register_full_backward_hook(bh)
    ratios=[.05,.1,.2,.3]; methods=["vanilla","vanilla7","center_gaussian","gradcam"]; baselines=["gray","black","blur"]
    fields=["style","video","frame","method","baseline"]+[f"ad{x}" for x in (5,10,20,30)]+[f"ai{x}" for x in (5,10,20,30)]+["insertion","deletion","p0"]
    os.makedirs(os.path.dirname(args.detail),exist_ok=True); rows=[]; seen=skipped=0
    with open(args.detail,"w",newline="",encoding="utf-8") as detail:
        writer=csv.DictWriter(detail,fieldnames=fields); writer.writeheader()
        for style in STYLES:
            ds=heldout(args.data_root,style,args.frames,args.videos); print(f"{style}: {len(ds)} videos",flush=True)
            for vi,path in enumerate(ds):
                video=os.path.basename(path)
                for t in range(args.frames):
                    rgb=cv2.cvtColor(cv2.imread(os.path.join(path,f"frame_{t}.jpg")),cv2.COLOR_BGR2RGB); frame=TFM(rgb).unsqueeze(0).to(device).requires_grad_(True)
                    logits=model.forward_frame(frame); seen+=1
                    if F.softmax(logits,1)[0,1].item()<.5: skipped+=1; continue
                    model.zero_grad(); logits[0,1].backward(); vanilla=normalize_map(frame.grad.abs().mean(1,keepdim=True)).detach()
                    gc=normalize_map(F.interpolate(F.relu((grads.detach().mean((2,3),keepdim=True)*acts.detach()).sum(1,keepdim=True)),(224,224),mode="bilinear",align_corners=False))
                    cams={"vanilla":vanilla,"vanilla7":resample(vanilla,7),"center_gaussian":priors(device,42)["center_gaussian"],"gradcam":gc}
                    for method,cam in cams.items():
                        for baseline in baselines:
                            row={"style":style,"video":video,"frame":t,"method":method,"baseline":baseline}; row.update(evaluate(model,frame.detach(),cam,baseline,ratios,args.steps)); writer.writerow(row); rows.append(row)
                detail.flush()
                if (vi+1)%5==0: print(f"  {vi+1}/{len(ds)}",flush=True)
    with open(args.output,"w",encoding="utf-8") as f:
        f.write(f"seen={seen},skipped={skipped},coverage={(seen-skipped)/seen:.6f}\n")
        f.write("method,baseline,n,ad5,ad10,ad20,ad30,ai5,ai10,ai20,ai30,insertion,deletion\n")
        for method in methods:
            for baseline in baselines:
                subset=[r for r in rows if r["method"]==method and r["baseline"]==baseline]
                vals=[method,baseline,str(len(subset))]+[f"{np.mean([r[k] for r in subset]):.6f}" for k in ["ad5","ad10","ad20","ad30","ai5","ai10","ai20","ai30","insertion","deletion"]]
                f.write(",".join(vals)+"\n")
    print(f"wrote {args.output} and {args.detail}",flush=True)
if __name__=="__main__": main()
