import argparse, csv, os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="./results/robustness_framewise.csv")
    p.add_argument("--output",default="./results/statistical_analysis.txt"); p.add_argument("--figdir",default="./figures"); p.add_argument("--bootstrap",type=int,default=2000); args=p.parse_args()
    with open(args.input,encoding="utf-8") as f: rows=list(csv.DictReader(f))
    numeric=["ad5","ad10","ad20","ad30","ai5","ai10","ai20","ai30","insertion","deletion","p0"]
    for r in rows:
        for k in numeric:r[k]=float(r[k])
        r["cluster"]=r["style"]+"/"+r["video"]
    methods=sorted(set(r["method"] for r in rows)); baselines=sorted(set(r["baseline"] for r in rows)); rng=np.random.default_rng(42)
    os.makedirs(args.figdir,exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as out:
        out.write("Cluster bootstrap by video (95% CI)\n")
        for baseline in baselines:
            out.write(f"\nBASELINE={baseline}\n")
            summaries=[]
            for method in methods:
                sub=[r for r in rows if r["method"]==method and r["baseline"]==baseline]; by=defaultdict(list)
                for r in sub:by[r["cluster"]].append(r)
                vm={c:{k:np.mean([x[k] for x in rs]) for k in ("ad10","insertion","deletion")} for c,rs in by.items()}; clusters=list(vm)
                cis={}
                for metric in ("ad10","insertion","deletion"):
                    boot=[]
                    for _ in range(args.bootstrap):
                        pick=rng.choice(clusters,len(clusters),replace=True); boot.append(np.mean([vm[c][metric] for c in pick]))
                    vals=np.array(boot); cis[metric]=(np.mean([vm[c][metric] for c in clusters]),np.quantile(vals,.025),np.quantile(vals,.975))
                summaries.append((method,cis)); out.write(method+" "+" ".join(f"{k}={v[0]:.6f}[{v[1]:.6f},{v[2]:.6f}]" for k,v in cis.items())+"\n")
            ranks={metric:{m:i+1 for i,(m,_) in enumerate(sorted(summaries,key=lambda x:x[1][metric][0],reverse=(metric=="insertion")))} for metric in ("ad10","insertion","deletion")}
            out.write("RANKS "+"; ".join(metric+":"+",".join(f"{m}={r}" for m,r in ranks[metric].items()) for metric in ranks)+"\n")
    colors={"vanilla":"#777777","vanilla7":"#4C78A8","center_gaussian":"#E45756","gradcam":"#54A24B"}; ratios=[5,10,20,30]
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=False)
    for ax,baseline in zip(axes,baselines):
        for method in methods:
            sub=[r for r in rows if r["method"]==method and r["baseline"]==baseline]; ax.plot(ratios,[np.mean([r[f"ad{x}"] for r in sub]) for x in ratios],marker="o",label=method,color=colors.get(method))
        ax.set_title(baseline); ax.set_xlabel("Kept pixels (%)"); ax.set_ylabel("AD (%)"); ax.grid(alpha=.25)
    axes[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(args.figdir,"keep_ratio_robustness.png"),dpi=220); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,baseline in zip(axes,baselines):
        for method in methods:
            sub=[r for r in rows if r["method"]==method and r["baseline"]==baseline]; ax.scatter(np.mean([r["ad10"] for r in sub]),np.mean([r["deletion"] for r in sub]),s=65,label=method,color=colors.get(method)); ax.annotate(method,(np.mean([r["ad10"] for r in sub]),np.mean([r["deletion"] for r in sub])),fontsize=7)
        ax.set_title(baseline); ax.set_xlabel("AD@10 (%)"); ax.set_ylabel("Deletion AUC"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(args.figdir,"ad_deletion_disagreement.png"),dpi=220); plt.close(fig)
if __name__=="__main__":main()
