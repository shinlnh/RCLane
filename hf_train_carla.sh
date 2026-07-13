#!/usr/bin/env bash
#
# Train RCLane on CARLA via HuggingFace Jobs (runs on HF GPU infra, not this machine).
#
# Pipeline inside the job:
#   1. install deps (torch comes from the image; add shapely + opencv + hf cli)
#   2. pull the training code from the HF Space repo (BanVienCorp/RCLane_2D_Detection)
#   3. download the dataset zip from the HF dataset repo (feat branch revision)
#   4. extract it  -> data_root = <extract>/data/dataset  (verified layout)
#   5. train.py --dataset carla, with F1 validation on label_val.json and
#      per-epoch checkpoint upload to the model repo (--push-to)
#
# The training loop itself uploads each epoch's checkpoint (best/last/e<N>) to the
# model repo as it goes, so if the job is interrupted you keep every finished epoch
# and can restart with --resume <repo>/last.pth.
#
# PREREQUISITE: log in first so --secrets HF_TOKEN can forward your token, and push
# the training code to the Space so the job can pull it:
#   hf auth login
#   (push code to BanVienCorp/RCLane_2D_Detection -- see below)
#
# Then:  bash hf_train_carla.sh
# Watch: hf jobs logs <JOB_ID>   (the script prints the id; it runs detached)

set -uo pipefail

# ---- knobs you may want to tweak -------------------------------------------
FLAVOR="l4x1"          # GPU: l4x1 (L4 24GB, cheap) | a10g-large | a100-large ...
TIMEOUT="12h"          # hard cap on job runtime
IMAGE="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
CODE_SPACE="BanVienCorp/RCLane_2D_Detection"   # HF Space holding the training code
VISION="b0"            # RCLane-S; b1/b2 for bigger
EPOCHS=20
BATCH=32
WORKERS=4              # if you hit a DataLoader "bus error", drop to 0 or 2
CKPT_REPO="BanVienCorp/LaneATT-Carla-checkpoints"
CKPT_SUBDIR="carla-${VISION}"   # folder inside the model repo
EVAL_EVERY=1           # run F1 validation every N epochs
EVAL_SUBSET="--eval-subset 500"  # cap val images per eval for speed; "" = full val set
# Resume a previous run in a NEW job: the container is fresh, so first pull last.pth
# from the model repo, then point --resume at it. Set RESUME_FROM_HUB=1 to do that.
RESUME_FROM_HUB=0
RESUME=""
# For a quick pipeline smoke test first, set: SUBSET="--subset 64" and EPOCHS=1
SUBSET=""
# ----------------------------------------------------------------------------

JOB_SCRIPT=$(cat <<EOF
set -eux

pip install -q shapely opencv-python-headless "huggingface_hub[cli]"

# 1) training code -- pull the HF Space repo (private -> uses HF_TOKEN from env)
hf download ${CODE_SPACE} --repo-type space --local-dir /workspace/code
cd /workspace/code

# 2) dataset zip from the HF dataset repo (the feat-branch revision with the clean zip)
hf download BanVienCorp/dataset_laneatt_fullmap \
    --repo-type dataset \
    --revision feat/add-dataset-laneatt-fulltown-clean \
    dataset_laneatt_fulltown_clean.zip \
    --local-dir /workspace/dl

# 3) extract -> images under data/dataset, labels at data/dataset/label_{train,val}.json
python -c "import zipfile; zipfile.ZipFile('/workspace/dl/dataset_laneatt_fulltown_clean.zip').extractall('/workspace/ds')"

# 3b) (optional) pull last.pth from the model repo to resume a previous run
if [ "${RESUME_FROM_HUB}" = "1" ]; then
    mkdir -p /workspace/ckpt
    hf download ${CKPT_REPO} --repo-type model \
        ${CKPT_SUBDIR}/last.pth --local-dir /workspace/resume
    cp /workspace/resume/${CKPT_SUBDIR}/last.pth /workspace/ckpt/last.pth
    RESUME_ARG="--resume /workspace/ckpt/last.pth"
else
    RESUME_ARG="${RESUME}"
fi

# 4) train (data_root MUST point at data/dataset). train.py uploads each epoch's
#    checkpoint to the model repo itself via --push-to, so nothing is lost mid-run.
python train.py --dataset carla \
    --data-root /workspace/ds/data/dataset \
    --label label_train.json \
    --eval-list label_val.json --eval-f1 --eval-every ${EVAL_EVERY} ${EVAL_SUBSET} \
    --vision ${VISION} --epochs ${EPOCHS} --batch ${BATCH} \
    --workers ${WORKERS} --amp --device cuda \
    --out /workspace/ckpt \
    --push-to ${CKPT_REPO} --push-subdir ${CKPT_SUBDIR} \
    \${RESUME_ARG} ${SUBSET}

echo "DONE -> https://huggingface.co/${CKPT_REPO}/tree/main/${CKPT_SUBDIR}"
EOF
)

hf jobs run \
    --flavor "${FLAVOR}" \
    --timeout "${TIMEOUT}" \
    --secrets HF_TOKEN \
    --detach \
    "${IMAGE}" \
    bash -c "${JOB_SCRIPT}"
