# Resolution and Spatial Priors Bias Deepfake CAM Faithfulness

Reproduction code for the ResNet-50 experiments and ViT-B/16 cross-backbone
validation reported in the MMM 2027 paper.

## Repository contents

- `benchmarks/`: ResNet-50 experiments for resolution controls, spatial-prior
  controls, CAM methods, perturbation robustness and Celeb-DF v2.
- `model_cnn.py`: ResNet-50 detector definition used by the main experiments.
- `supplement_experiments_logit_v2.py`: SmoothGrad and Integrated Gradients
  experiments, including 7x7 resolution controls, on fake frames.
- `vit_xai/`: ViT-B/16 training and cross-backbone explanation evaluation.
- `results/`: paper result tables, frame-wise CSVs and the fake-frame
  SmoothGrad / Integrated Gradients supplement.

The main paper experiments use the ResNet-50 CNN. ViT-B/16 is used only for
cross-backbone validation.

## Environment

Use Python 3.11. Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The reported environment used PyTorch 2.6.0 and torchvision 0.21.0 with CUDA
12.4, NumPy 2.0.2, OpenCV 4.12.0.88, scikit-learn 1.6.1, matplotlib 3.9.4
and Pillow 11.3.0. The requirements file selects the CUDA 12.4 PyTorch wheels;
for CPU-only installation, remove the CUDA wheel index and install the
appropriate PyTorch build for your platform.

## Data and checkpoints

Datasets and model checkpoints are not included. They must be obtained and
prepared separately:

- FaceForensics++ c23, using the eight-frame extracts expected by the scripts.
- Celeb-DF v2 for the cross-dataset experiment.
- The ResNet-50 epoch-17 checkpoint and ViT-B/16 epoch-13 checkpoint.

The programs expect the dataset folders to contain one directory per video and
frames named `frame_0.jpg` through `frame_7.jpg`. Set the data and checkpoint
locations and thread limits before running an experiment:

For the FF++ experiments, `DATA_ROOT` should have this layout (ViT training
also uses `FaceSwap`, `NeuralTextures` and `original`):

```text
FFPlusPlus_Extracted_8Frames/
  Deepfakes/<video_id>/frame_0.jpg ... frame_7.jpg
  Face2Face/<video_id>/frame_0.jpg ... frame_7.jpg
  FaceShifter/<video_id>/frame_0.jpg ... frame_7.jpg
  DeepFakeDetection/<video_id>/frame_0.jpg ... frame_7.jpg
  FaceSwap/<video_id>/frame_0.jpg ... frame_7.jpg
  NeuralTextures/<video_id>/frame_0.jpg ... frame_7.jpg
  original/<video_id>/frame_0.jpg ... frame_7.jpg
```

```bash
export DATA_ROOT=/path/to/FFPlusPlus_Extracted_8Frames
export CHECKPOINT="$PWD/checkpoints/resnet50_best.pt"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
mkdir -p logs results/benchmarks results/smoothgrad_ig
```

On Linux, run each experiment separately with `nohup` and `taskset` to keep CPU
use to four cores. For example, the controlled-resolution experiment is:

```bash
nohup taskset -c 0-3 python -u benchmarks/controlled_benchmark.py \
  --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" \
  --max-videos 40 --frames 8 --steps 10 \
  --output results/benchmarks/controlled_benchmark.txt \
  > logs/controlled.log 2>&1 < /dev/null & echo $! > logs/controlled.pid
```

The other ResNet-50 experiments use the same environment:

```bash
# Spatial-prior controls
nohup taskset -c 0-3 python -u benchmarks/prior_attack_benchmark.py \
  --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" \
  --frames 8 --max-videos 40 --steps 10 \
  --output results/benchmarks/prior_attack_benchmark.txt \
  > logs/prior_attack.log 2>&1 < /dev/null & echo $! > logs/prior_attack.pid

# Five CAM families
nohup taskset -c 0-3 python -u benchmarks/cam_method_benchmark.py \
  --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" \
  --frames 8 --max-videos 20 --steps 5 --score-topk 32 \
  --output results/benchmarks/cam_method_benchmark.txt \
  > logs/cam_methods.log 2>&1 < /dev/null & echo $! > logs/cam_methods.pid

# Keep-ratio / fill robustness and frame-wise results
nohup taskset -c 0-3 python -u benchmarks/robustness_benchmark.py \
  --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" \
  --frames 8 --videos 20 --steps 5 \
  --detail results/benchmarks/robustness_framewise.csv \
  --output results/benchmarks/robustness_benchmark.txt \
  > logs/robustness.log 2>&1 < /dev/null & echo $! > logs/robustness.pid
```

After the robustness run has finished, regenerate its bootstrap analysis and
qualitative figure:

```bash
mkdir -p figures
nohup taskset -c 0-3 python -u benchmarks/analyze_robustness.py \
  --input results/benchmarks/robustness_framewise.csv \
  --output results/benchmarks/statistical_analysis.txt --figdir figures \
  > logs/statistical_analysis.log 2>&1 < /dev/null & echo $! > logs/statistical_analysis.pid
```

After the statistical analysis has finished:

```bash
nohup taskset -c 0-3 python -u benchmarks/make_evidence_figures.py \
  --csv results/benchmarks/robustness_framewise.csv \
  --data-root "$DATA_ROOT" --checkpoint "$CHECKPOINT" \
  --output figures/qualitative_counterexamples.png \
  > logs/evidence_figures.log 2>&1 < /dev/null & echo $! > logs/evidence_figures.pid
```

Celeb-DF uses a separate extracted-frame root:

```bash
export CELEBDF_ROOT=/path/to/Celeb-DF-v2-extracted-frames
nohup taskset -c 0-3 python -u benchmarks/celebdf_bias_benchmark.py \
  --data-root "$CELEBDF_ROOT" --checkpoint "$CHECKPOINT" \
  --videos 80 --frames 8 --steps 10 \
  --output results/benchmarks/celebdf_bias_benchmark.txt \
  > logs/celebdf.log 2>&1 < /dev/null & echo $! > logs/celebdf.pid
```

`CELEBDF_ROOT` points directly to the `Celeb-synthesis` directory containing
the video subdirectories, not its parent dataset directory.

## Main experiment scripts

The CNN experiments are implemented in `benchmarks/`:

- `controlled_benchmark.py`: resolution and smoothing controls.
- `prior_attack_benchmark.py`: spatial-prior controls.
- `cam_method_benchmark.py`: comparison of CAM methods.
- `robustness_benchmark.py`: keep-ratio and perturbation-fill robustness.
- `celebdf_bias_benchmark.py`: cross-dataset evaluation on Celeb-DF v2.
- `make_evidence_figures.py` and `analyze_robustness.py`: evidence figures and
  supporting statistical analysis.

The ViT experiment is implemented in `vit_xai/vit_xai_benchmark.py`; its model
training code is in `vit_xai/train_vit.py`.

The checked-in results map to the experiments as follows:

| Result file | Experiment |
|---|---|
| `results/benchmarks/controlled_benchmark.txt` | Resolution and smoothing controls |
| `results/benchmarks/prior_attack_benchmark.txt` | Spatial-prior controls |
| `results/benchmarks/cam_method_benchmark.txt` | Five CAM methods |
| `results/benchmarks/robustness_benchmark.txt` and `robustness_framewise.csv` | Keep-ratio / fill robustness |
| `results/benchmarks/statistical_analysis.txt` | Video-level bootstrap analysis |
| `results/benchmarks/celebdf_bias_benchmark.txt` | Celeb-DF v2 evaluation |
| `results/smoothgrad_ig/fake.csv` | SmoothGrad / Integrated Gradients fake-frame experiment |
| `results/benchmarks/vit_xai_benchmark.txt` and `vit_xai_framewise.csv` | ViT cross-backbone evaluation |

Run the Table 2 fake-frame experiment with:

```bash
nohup taskset -c 0-3 python -u supplement_experiments_logit_v2.py \
  --data-root /path/to/FFPlusPlus_Extracted_8Frames \
  --checkpoint /path/to/resnet50_best.pt \
  --insertion-steps 5 \
  --output results/smoothgrad_ig/fake.csv \
  > logs/smoothgrad_ig.log 2>&1 < /dev/null & echo $! > logs/smoothgrad_ig.pid
```

The experiment uses 224x224 inputs, ImageNet normalization, class-logit
attribution, a 10% top-k mask and black fill for AD. Insertion/deletion use a
gray fill (pixel value 0.5). SmoothGrad uses 30 samples with sigma 0.1 in
normalized input space; Integrated Gradients uses a black baseline and 32
steps. The reported run used seed 42 and four CPU threads.

For the supplementary ViT cross-backbone experiment, train the model and then
run the explanation benchmark:

```bash
nohup taskset -c 0-3 python -u vit_xai/train_vit.py \
  --data-root "$DATA_ROOT" --output-dir checkpoints/vit \
  --epochs 20 --batch-size 8 --num-workers 4 --cpu-cores 4 \
  > logs/vit_train.log 2>&1 < /dev/null & echo $! > logs/vit_train.pid
```

After training completes:

```bash
nohup taskset -c 0-3 python -u vit_xai/vit_xai_benchmark.py \
  --data-root "$DATA_ROOT" --checkpoint checkpoints/vit/best.pt \
  --frames 8 --max-videos 20 --steps 5 \
  --output results/benchmarks/vit_xai_benchmark.txt \
  --detail results/benchmarks/vit_xai_framewise.csv \
  > logs/vit_xai.log 2>&1 < /dev/null & echo $! > logs/vit_xai.pid
```

The checked-in result files record the outputs used for the paper. Re-running
an experiment requires the matching dataset split and checkpoint; aggregate
values can differ if the input data, checkpoint or evaluation parameters differ.

## Citation

```bibtex
@inproceedings{zhang2027resolution,
  title     = {Resolution and Spatial Priors Bias Deepfake CAM Faithfulness},
  author    = {Zhang, Yan and Li, Xiaojie and Yang, Fan},
  booktitle = {33rd International Conference on Multimedia Modeling (MMM)},
  year      = {2027}
}
```

