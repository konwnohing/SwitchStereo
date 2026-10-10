# SwitchStereo: Stereo Matching with Switchable Refinement for Low-Power Embedded GPUs

Official implementation of the paper

> **SwitchStereo: Stereo Matching with Switchable Refinement for Low-Power Embedded GPUs**
> Lihong Lu, Song Wang, Keke Shang. Submitted to *The Visual Computer*.
> (The network was called *ESNet* in earlier versions of this repository.)

SwitchStereo combines a RAFT-Stereo-style recurrent correlation matcher (RCM) at 1/8 resolution with a progressive refinement network (PRN) and a confidence-gated refinement module (Gate). It is deployed with TensorRT FP16 on a 4 GB NVIDIA Jetson Nano as two engines:

* **Fast**: feature extractor + RCM, bilinear up-sampling (separately trained model without PRN and Gate);
* **Balanced**: feature extractor + RCM + PRN of the full model, with the branches that the trained network no longer uses removed from the exported graph.

The repository contains the training code, the configurations of every model in the paper, the scripts that reproduce all tables, and the export, branch-removal and Jetson benchmarking scripts.

## Results

KITTI 2015 benchmark (test set): **D1-all 3.38%** (D1-bg 2.93%, D1-fg 5.63%).

Held-out KITTI 2015 pairs 000121–000199 (79 pairs) and Jetson Nano 4 GB (TensorRT 8.2.1, FP16, 320×640, MAXN):

| Engine | Model | D1-all, crop (%) | D1-all, full (%) | Latency (ms) | FPS | Energy (J/frame) | TensorRT memory (MiB) |
|---|---|---|---|---|---|---|---|
| Fast | model A | 3.92 | 4.25 | 143.5 | 6.97 | 0.95 | 155 |
| Balanced | final model, inactive branches removed | 2.88 | 3.69 | 410.8 | 2.43 | 2.64 | 203 |
| Balanced (for comparison) | final model, inactive branches kept | 2.89 | 3.69 | 1621.2 | 0.62 | 9.43 | 506 |

Crop: 384×1152 centre crop; full: whole image. See the paper for the ablations, boundary and depth analyses and the per-layer latency profile.

## Repository structure

```text
SwitchStereo/
├── configs/
│   ├── paper/              # exact configurations of all models in the paper (exported from the checkpoints)
│   └── ...                 # original working configurations
├── data/                   # dataset loaders and transforms
├── engines/                # trainer and evaluator
├── losses/                 # multi-scale OHEM-L1 loss and auxiliary terms
├── models/                 # SwitchStereo architecture
├── scripts/
│   ├── train_main.py       # training entry (models A, B, C, OHEM, final, Scene Flow pre-training)
│   ├── train_ohem_ratio.py # training entry of the OHEM-ratio experiment (Table 5)
│   ├── eval_ohem.py        # evaluation of the OHEM-ratio models (Table 5)
│   ├── merge_kitti.py      # merges KITTI 2012 and 2015 into one training folder
│   └── submit_kitti.py     # KITTI benchmark submission
├── analysis/               # scripts that reproduce the tables and figures of the paper
├── deploy/                 # ONNX fixes, branch removal, consistency check and Jetson scripts
├── requirements.txt
└── README.md
```

## Installation

```bash
conda create -n switchstereo python=3.10 -y
conda activate switchstereo
pip install -r requirements.txt
pip install onnx onnxruntime        # only needed for the deployment scripts
```

Deployment on the Jetson Nano uses JetPack 4.6.1 (L4T R32.7.1, CUDA 10.2, TensorRT 8.2.1) and the MMDeploy TensorRT plugin library `libmmdeploy_tensorrt_ops.so` for grid sampling.

## Data

1. Download [KITTI 2015 and KITTI 2012 stereo](https://www.cvlibs.net/datasets/kitti/) (and the KITTI 2015 calibration files for the depth analysis), and [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) for pre-training.
2. Merge KITTI 2012 and 2015 into one folder (edit the paths at the top of the script):
   ```bash
   python scripts/merge_kitti.py      # -> datasets/kitti_mix/training/{image_2,image_3,disp_occ_0}
   ```
   Files are renamed `2012_*` and `2015_*`. After sorting, the first 80% (315 pairs: all KITTI 2012 pairs and KITTI 2015 000000–000120) are used for training and the last 20% (KITTI 2015 000121–000199, 79 pairs) are held out.
3. Put the KITTI 2015 calibration files in `datasets/kitti2015/training/calib_cam_to_cam/` if you want to reproduce the depth table.

## Pretrained weights

Download the weights from the [Releases](../../releases) page and keep the folder names below, which the evaluation scripts expect:

| Folder | Model in the paper |
|---|---|
| `checkpoints_sceneflow_pretrain_1/` | Scene Flow pre-training (zero-shot result, initialization of all KITTI models) |
| `checkpoints_modelA/` | model A, no PRN and no Gate — **Fast engine** |
| `checkpoints_modelrpn/` | model B, PRN |
| `checkpoints_baseline/` | model C, PRN + Gate, L1 |
| `checkpoints_ohem/` | PRN + Gate, OHEM-L1 |
| `checkpoints_ohem+spatial/` | final model — **Balanced engine** |
| `checkpoints_ohem_ablation/rho_0p2, rho_0p3, rho_0p4/` | OHEM-ratio experiment (Table 5) |

Each folder contains `model_best.pth`.

## Reproducing the tables

All accuracy tables use the 79 held-out pairs.

```bash
# Tables 3, 4, 6, 7, 8 and 12 (crop, full image, boundary, inference-time variants, depth)
python analysis/reproduce_tables.py --proj . --kitti_root datasets/kitti2015 \
    --extra checkpoints_modelA/model_best.pth:raw_bilinear:ModelA

# Table 5 (OHEM ratio; 320×640 crop on KITTI 2015 000160–000199)
python scripts/eval_ohem.py --config configs/ohem_ablation.yaml --resume checkpoints_ohem_ablation/rho_0p3/model_best.pth

# Gate analysis (correction applied by the Gate at inference) and the training settings of each model
python analysis/check_gate.py --proj . --kitti_root datasets/kitti2015

# Inference-time variants and effective MACs of the final model
python analysis/eval_variants.py --kitti_root datasets/kitti2015 --ckpts checkpoints_ohem+spatial/model_best.pth

# Fig. 3
python analysis/make_fig3.py --ckpt checkpoints_ohem+spatial/model_best.pth --kitti_root datasets/kitti2015
```

Run the scripts from the repository root, because they import `models/` from the current directory.

## Training

```bash
# Scene Flow pre-training (ImageNet-initialized ResNet-18)
python scripts/train_main.py --config configs/paper/sceneflow_pretrain.yaml

# KITTI fine-tuning, e.g. the final model (use model_A.yaml, model_B.yaml, model_C.yaml or model_ohem.yaml for the ablations)
python scripts/train_main.py --config configs/paper/final.yaml \
    --resume checkpoints_sceneflow_pretrain_1/model_best.pth

# continue an interrupted run
python scripts/train_main.py --config configs/paper/final.yaml --continue_run
```

Fine-tuning runs for 300 epochs with batch size 4, 384×1152 crops and a cosine learning-rate schedule starting at 2×10⁻⁵; the checkpoint with the lowest held-out D1-all is kept. Edit `data.root_dir` in the configuration files to point to your data.

## Deployment

The ONNX graphs exported from the trained models are provided in the Releases. Starting from them:

```bash
# 1. set the grid-sampling plugin attributes to the training settings (align_corners = 1; zero padding in the RCM, border padding in the PRN)
python deploy/fix_onnx_grid_attrs.py

# 2. remove the inactive branches (Balanced engine) and build the shared-weight Fast variant
python deploy/make_onnx_variants.py

# 3. reference inputs and PyTorch outputs for the consistency check
python deploy/make_consistency_inputs.py --ckpt checkpoints_ohem+spatial/model_best.pth --kitti_root datasets/kitti2015

# 4. Fast engine with the weights of model A (graph surgery + weight swap, verified with ONNX Runtime)
python deploy/make_fast_modelA.py --ref checkpoints_ohem+spatial/model_best.pth --new checkpoints_modelA/model_best.pth
```

On the Jetson Nano, copy the resulting ONNX files, `consistency_inputs/`, the plugin library and the scripts in `deploy/` to one folder and run

```bash
bash jetson_final.sh          # builds the engines, measures latency, power and memory, runs the consistency check
bash jetson_fastA.sh          # the same for the Fast engine of model A
python3 compare_consistency.py
```

**Note.** Exporting grid sampling through the plugin with the default attributes (`align_corners = 0`, zero padding) silently changes the sampling positions; in our case it nearly doubled D1-all. Always compare the engine outputs with the PyTorch model before reporting deployment results.

## Citation

```bibtex
@article{switchstereo,
  title   = {SwitchStereo: Stereo Matching with Switchable Refinement for Low-Power Embedded GPUs},
  author  = {Lu, Lihong and Wang, Song and Shang, Keke},
  journal = {The Visual Computer},
  note    = {under review},
  year    = {2026}
}
```

## Acknowledgements

The matcher follows [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo); grid sampling on TensorRT uses the plugins of [MMDeploy](https://github.com/open-mmlab/mmdeploy).

## License

To be added.
