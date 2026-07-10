# RCLane (PyTorch)

A faithful PyTorch reimplementation of **RCLane** — relay-chain lane detection
(ECCV 2022) — ported from the original MindSpore code
([lpplbiubiubiub/RCLane](https://github.com/lpplbiubiubiub/RCLane)).

RCLane detects lanes by predicting, at every foreground pixel, a **transfer vector**
(the step to the next relay point) and a **distance** (steps left to the lane end),
then crawling those relay chains forward and backward into full lanes. This handles
curves, Y-shapes and near-horizontal lanes that row-anchor methods cannot.

## Branch map

This repo is organised so the shared network lives in one place and each dataset is
an isolated add-on.

| Branch | Purpose |
|---|---|
| `main` | Project overview + license (this page). |
| `dev` | **Integration branch** — the full port plus all dataset loaders. Trains any dataset via `--dataset carla\|culane\|curvelanes`. Start here. |
| `feat/train-RCLane-with-CARLA` | Foundation + the CARLA loader only. **Primary target.** |
| `feat/train-RCLane-with-CULane` | Foundation + the CULane loader only. Used to check performance against the paper (RCLane-S 79.5 F1). |
| `feat/train-RCLane-with-CurveLanes` | Foundation + the CurveLanes loader only. |

Every branch shares a common **foundation** commit — the dataset-agnostic port:
`rclane.py` (SegFormer backbone + head), `loss.py`, `encode.py`/`decode.py` (the
relay-chain codec), `dataset.py` (base), and `train.py` (the training loop with a
`--dataset` dispatcher). Each feature branch adds a single `dataset_<name>.py`.

## What the project does

- **Goal:** train RCLane on CARLA-simulated lane data (the primary target), and
  reproduce the paper's CULane / CurveLanes numbers to validate the port.
- **Status:** network, loss, encode/decode codec, datasets and training loop are done
  and verified end-to-end (an 8-image overfit drives the loss 42.5 → 7.2). Still to do:
  data augmentation, F1 evaluation, and a full GPU training run.

## Datasets

The network is dataset-agnostic; each branch trains against one of the datasets below.
Download the one matching your target branch and point `--data-root` at its extracted
folder.

| Dataset | Used by | Download |
|---|---|---|
| **CULane** | `feat/train-RCLane-with-CULane` | [xingangpan.github.io/projects/CULane.html](https://xingangpan.github.io/projects/CULane.html) — official release (Google Drive, ~50 GB). Grab every `driver_*` archive **plus** `laneseg_label_w16.tar.gz` and `list.tar.gz`. |
| **CurveLanes** | `feat/train-RCLane-with-CurveLanes` | [github.com/SoulmateB/CurveLanes](https://github.com/SoulmateB/CurveLanes) — official release. |
| **CARLA** (custom) | `feat/train-RCLane-with-CARLA` | [huggingface.co/datasets/BanVienCorp/dataset_laneatt_fullmap](https://huggingface.co/datasets/BanVienCorp/dataset_laneatt_fullmap/tree/feat%2Fadd-dataset-laneatt-fulltown-clean) — CARLA-simulated lane data (`feat/add-dataset-laneatt-fulltown-clean` branch). |

## Getting started

```bash
git checkout dev
python rclane.py   # sanity: forward pass + parameter counts vs the paper
python train.py --dataset carla --data-root <CARLA_ROOT> --device cuda
```

See the README on `dev` for full technical details, validation numbers, and per-dataset
run commands.

## Credits

Method: *RCLane: Relay Chain Prediction for Lane Detection*, Xu et al., ECCV 2022.
Original MindSpore implementation by lpplbiubiubiub. Licensed under MIT (see `LICENSE`).
