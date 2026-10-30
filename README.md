# Fused3DDetection

Multi-modal (LiDAR + multi-view camera) 3D object detection for autonomous driving, built as a plugin for [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) (1.x / `mmengine` API) and evaluated on [nuScenes](https://www.nuscenes.org/).

The detector (`MFDetector`) follows a TransFusion-style query-based design: object queries are initialised from a LiDAR BEV heatmap and then refined by transformer decoders. The decoders attend jointly to LiDAR BEV tokens and to image feature tokens, and both token types share one BEV positional embedding space.

## Method overview

```
 LiDAR sweeps ──► SPConv voxelization ──► HardSimpleVFE ──► SparseEncoder ──► SECOND + SECONDFPN ──► BEV feature (512 ch)
                                                                                                         │
 6 camera images ──► image backbone (EVA-ViT / Swin-T / VoVNet-99 / R50) ──► FPN ──► image tokens       │
                                                                                     │                  │
                     LiDAR points projected into each image (dataloader) ─► points_map (x, y, z per pixel)
                                                                                     │                  ▼
                                                                                     │   MFFusionHead
                                                                                     │   ├─ BEV heatmap → top-K queries (+ class embedding)
                                                                                     └──►├─ Center decoder: queries × [BEV tokens ∪ image tokens]
                                                                                         │   → center offset refinement
                                                                                         ├─ Decoupled decoder: box branch / cls branch
                                                                                         └─ Prediction heads: center, height, dim, rot, vel, heatmap
```

Main ideas in the code:

- **Heatmap query initialisation.** A dense class heatmap is predicted on the BEV map. After local-max NMS (kernel 3; no NMS for pedestrian and traffic cone on nuScenes), the top `num_proposals` (200) peaks become queries. Their features are the BEV feature at the peak plus a one-hot class encoding.
- **Multi-stage heatmap** (`multistage_heatmap > 0`). Extra heatmap heads pick more proposals while masking out locations already selected, so the query set grows to `num_proposals × (stages + 1)`.
- **Shared BEV positional space for image tokens.** `MFMultiViewImageFeatureCorrespondingPoints` (`projects/mmdet3d_plugin/model/loading.py`) projects LiDAR points into every camera at feature-map resolution (`downsample`). For each feature pixel it keeps the nearest point and stores its 3D location (`points_map`) and a validity mask (`points_map_mask`). Image tokens then get the same sine and MLP positional embedding as BEV tokens, built from these xy coordinates. LiDAR and image tokens are concatenated into one key/value set.
- **Two-step decoding.** `CenterTransformerDecoderLayer` refines query centres first. `MFTransformerDecoderLayer` then runs separate cross-attention and FFN branches for box regression and classification, using FlashAttention through `PETRMultiheadFlashAttention`.
- **Query denoising.** Noised GT boxes (`add_gt_groups`, `noise_scale`, `noise_trans`) are added as extra queries during training to make training more stable.
- **Training targets and losses.** Hungarian assignment uses focal, BEV-L1 and 3D-IoU costs. The losses are focal (cls), Gaussian focal (heatmap), L1 (bbox) and 3D IoU. `RDIoULoss` and `SoftFocalLoss` are available as alternatives.

## Repository layout

```
projects/
├── configs/
│   ├── lidar/
│   │   ├── lidar.py                      # LiDAR-only baseline (full model/data/schedule definition)
│   │   └── lidar_multistage_heatmap.py   # LiDAR-only + multi-stage heatmap proposals
│   └── fusion/
│       ├── fusion_evavit.py              # EVA-ViT-L (frozen, flash-attn, SimpleFPN), 320×800 input
│       ├── fusion_SwinTransformer.py     # Swin-T + GeneralizedLSSFPN
│       ├── fusion_vov.py                 # VoVNet-99-eSE + CPFPN, 640×1600 input
│       └── fusion_r50.py                 # ResNet-50 + CPFPN (inherits lidar/lidar.py)
├── mmdet3d_plugin/model/
│   ├── MFFusion.py           # MFDetector: voxelization, LiDAR/image feature extraction, loss/predict
│   ├── MFFusion_head.py      # MFFusionHead: heatmap queries, query denoising, decoders, targets, losses
│   ├── transformer.py        # Center / decoupled decoder layers, flash-attention MHA
│   ├── attention.py          # FlashAttention / FlashMHA wrappers
│   ├── prediction_head.py    # MFSeparateHead (separate box / cls regression branches)
│   ├── utils.py              # TransFusionBBoxCoder, Hungarian/Heuristic assigners, match costs
│   ├── rdiouloss.py          # IoU3DLoss, RDIoULoss, SoftFocalLoss
│   ├── loading.py            # Multi-view image loading, LiDAR→image correspondence maps
│   ├── transforms_3d.py      # ImageAug3D, synchronized LiDAR/camera flip & rot-scale-trans, GridMask
│   ├── eva_vit.py            # EVA-ViT backbone + simple feature pyramid
│   ├── vovnetcp.py, cp_fpn.py, bevfusion_necks.py   # other image backbones / necks
│   ├── sparse_encoder.py     # BEVFusion-style sparse encoder
│   ├── point_sa_module2.py   # point set-abstraction module (experimental)
│   ├── nuscenes_metrics.py   # MFNuScenesMetric
│   └── ops/voxel/            # SPConv-based voxelization
└── setup.py                  # builds CUDA extensions (see note below)
tools/
├── train.py / test.py        # mmengine Runner entry points
├── dist_train.sh / dist_test.sh
├── create_data.py / create_data.sh, data_converter/   # nuScenes info + GT-database generation
├── test_speed.py
└── analysis_tools/           # log analysis, benchmark, FLOPs
```

## Requirements

The code targets Linux with CUDA GPUs (the prebuilt extensions in `build/` are for Python 3.8, x86_64). Main dependencies:

- PyTorch (with CUDA), `mmengine`, `mmcv` 2.x, `mmdet` 3.x, `mmdet3d` 1.x
- `spconv` (voxelization), `torch_scatter`
- `flash_attn` (transformer decoder attention and EVA-ViT)
- `einops`, `fvcore`, `scipy`, `nuscenes-devkit`

Install mmdet3d by following its [official guide](https://mmdetection3d.readthedocs.io/en/latest/get_started.html), then install the other packages with pip.

> **CUDA extensions.** `projects/setup.py` currently only lists a `point_utils` extension under `projects/mmdet3d_plugin/model/PointUtils`. The `bev_pool` and `voxel` extensions are commented out. Those sources are not part of this repository. The models in the provided configs use `SPConvVoxelization` and mmcv ops, so they do not need these extensions.

## Data preparation

Download nuScenes (v1.0 trainval, and test if needed) and use the standard layout:

```
data/nuscenes/
├── maps/  samples/  sweeps/  v1.0-trainval/  v1.0-test/
```

Generate info files and the GT database used for `ObjectSample` augmentation:

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```

This creates `nuscenes_infos_{train,val}.pkl` and `nuscenes_dbinfos_train.pkl`.

## Configuration

Every config except `fusion_r50.py` inherits from a local `varibles.py` (`varibles_multistage_heatmap.py` for the multi-stage config) in the same directory. These files are listed in `.gitignore`, so you need to create them. They must define:

| Key | Meaning |
|---|---|
| `data_root` | path to nuScenes, e.g. `'data/nuscenes/'` |
| `class_names` | the 10 nuScenes detection classes |
| `metadata_version` | e.g. `'v1.0-trainval'` |
| `train_batch_size` | samples per GPU |
| `db_sampler` | GT-sampling config (see `projects/configs/lidar/lidar.py`) |
| `json_dir` | output prefix for result JSON files |
| `load_from` | checkpoint to initialise from (e.g. a LiDAR-only model) |
| `wandb_project`, `wandb_group` | W&B logging names |
| `custom_hooks` | optional extra hooks |

For the fusion configs, the usual pattern is for `varibles.py` to inherit from `../lidar/lidar.py` (which holds the full LiDAR branch, head, and dataset definitions) and set the keys above. Fusion training then starts from a pretrained LiDAR-only checkpoint via `load_from`. In the fusion configs, `ObjectSample` is turned off after epoch 4 (`DisableObjectSampleHook`).

The `downsample` of `MFMultiViewImageFeatureCorrespondingPoints` must equal the stride of the image feature map that the head uses (for example 4 for `fusion_evavit.py` at 320×800).

## Training

```bash
# single GPU
python tools/train.py projects/configs/lidar/lidar.py --work-dir work_dirs/lidar

# multi-GPU (8 GPUs)
bash tools/dist_train.sh projects/configs/lidar/lidar.py 8 --work-dir work_dirs/lidar

# fusion, initialised from the LiDAR checkpoint set via load_from
bash tools/dist_train.sh projects/configs/fusion/fusion_evavit.py 8 --work-dir work_dirs/fusion_evavit
```

Recommended order: train the LiDAR-only model first (20 epochs with a cyclic cosine schedule), then fine-tune a fusion config (about 6 epochs, AdamW, lr 1e-4).

`dist_train.sh` / `dist_test.sh` read the `NNODES`, `NODE_RANK`, `PORT` and `MASTER_ADDR` environment variables for multi-node runs.

## Evaluation

```bash
python tools/test.py <config> <checkpoint>
bash tools/dist_test.sh <config> <checkpoint> 8
```

Evaluation uses the nuScenes detection metrics (mAP, NDS) through `NuScenesMetric` / `MFNuScenesMetric`. Use `tools/test_speed.py` to measure inference speed and `tools/analysis_tools/get_flops.py` to count FLOPs.

## Acknowledgements

This project builds on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [TransFusion](https://github.com/XuyangBai/TransFusion), [BEVFusion](https://github.com/mit-han-lab/bevfusion), [PETR / StreamPETR](https://github.com/exiawsh/StreamPETR) (VoVNet, CPFPN, flash attention), [EVA](https://github.com/baaivision/EVA) and [DN-DETR](https://github.com/IDEA-Research/DN-DETR) (query denoising).
