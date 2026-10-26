import argparse,csv,os,sys
import cv2,numpy as np,torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
sys.path.insert(0,os.path.dirname(__file__))
from model_cnn import CNNBaseline
from controlled_benchmark import normalize_map,resample
from prior_attack_benchmark import priors
TFM=transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def main():
    p=argparse.ArgumentParser();p.add_argument("--csv",default="./results/robustness_framewise.csv");p.add_argument("--data-root",default="data/FFPlusPlus_Extracted_8Frames");p.add_argument("--checkpoint",default="./checkpoints/resnet50_best.pt");p.add_argument("--output",default="./figures/qualitative_counterexamples.png");args=p.parse_args()
    with open(args.csv,encoding="utf-8") as f:rows=list(csv.DictReader(f))
    gray=[r for r in rows if r["baseline"]=="gray"]; index={(r["style"],r["video"],r["frame"],r["method"]):r for r in gray}
    candidates=[]
    for r in gray:
        if r["method"]!="center_gaussian":continue
        g=index.get((r["style"],r["video"],r["frame"],"gradcam"));
        if g:candidates.append((float(g["deletion"])-float(r["deletion"]),r,g))
    selected=[]
    for style in ("Deepfakes","Face2Face","FaceShifter","DeepFakeDetection"):
        ss=[x for x in candidates if x[1]["style"]==style]; selected.append(sorted(ss,key=lambda x:(float(x[1]["ad10"]),-float(x[1]["deletion"])))[0])
    device=torch.device("cuda");model=CNNBaseline(num_classes=2,pretrained=False).to(device);ck=torch.load(args.checkpoint,map_location=device);model.load_state_dict(ck.get("model",ck));model.eval();acts=grads=None
    def fh(m,i,o):
        nonlocal acts;acts=o
    def bh(m,gi,go):
        nonlocal grads;grads=go[0]
    model.backbone.layer4.register_forward_hook(fh);model.backbone.layer4.register_full_backward_hook(bh)
    fig,axes=plt.subplots(4,5,figsize=(12,10)); names=["Input","Vanilla","Vanilla 7x7","Center prior","Grad-CAM"]
    for row,(score,meta,gmeta) in enumerate(selected):
        path=os.path.join(args.data_root,meta["style"],meta["video"],f"frame_{meta['frame']}.jpg");rgb=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB);frame=TFM(rgb).unsqueeze(0).to(device).requires_grad_(True);logits=model.forward_frame(frame);model.zero_grad();logits[0,1].backward()
        van=normalize_map(frame.grad.abs().mean(1,keepdim=True)).detach();gc=normalize_map(F.interpolate(F.relu((grads.detach().mean((2,3),keepdim=True)*acts.detach()).sum(1,keepdim=True)),(224,224),mode="bilinear",align_corners=False));cams=[None,van,resample(van,7),priors(device,42)["center_gaussian"],gc]
        for col,cam in enumerate(cams):
            axes[row,col].imshow(rgb)
            if cam is not None:axes[row,col].imshow(cam[0,0].cpu(),cmap="jet",alpha=.48,vmin=0,vmax=1)
            axes[row,col].axis("off");
            if row==0:axes[row,col].set_title(names[col])
        axes[row,0].set_ylabel(meta["style"],fontsize=9)
        for col,method in enumerate([None,"vanilla","vanilla7","center_gaussian","gradcam"]):
            if method:
                m=index[(meta["style"],meta["video"],meta["frame"],method)];axes[row,col].set_xlabel(f"AD={float(m['ad10']):.1f}  Del={float(m['deletion']):.3f}",fontsize=8)
    os.makedirs(os.path.dirname(args.output),exist_ok=True);fig.tight_layout();fig.savefig(args.output,dpi=220,bbox_inches="tight");plt.close(fig)
if __name__=="__main__":main()
