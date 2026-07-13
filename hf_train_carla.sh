#!/usr/bin/env bash
#
# Train RCLane on CARLA via HuggingFace Jobs (runs on HF GPU infra, not this machine).
#
# Pipeline inside the job:
#   1. install deps (torch comes from the image; add shapely + opencv + hf cli)
#   2. pull the training code from the HF Space repo (BanVienCorp/RCLane_2D_Detection)
#   3. download the dataset zip from the HF dataset repo (feat branch revision)
#   4. extract it  -> data_root = <extract>/data/dataset  (verified layout)
#   5. torchrun launches one DDP rank per H200; 42 DataLoader processes warm/feed
#      the GPUs and distributed F1 overlaps 8 loader + 36 decode processes
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
FLAVOR="${FLAVOR:-h200x2}"        # 2x H200 141GB, 46 vCPU, 512GB RAM
TIMEOUT="${TIMEOUT:-12h}"         # hard cap on job runtime
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
CODE_SPACE="${CODE_SPACE:-BanVienCorp/RCLane_2D_Detection}"
VISION="${VISION:-b0}"            # RCLane-S; b1/b2 for bigger
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-64}"              # per GPU; DDP global batch = 128
LR="${LR:-2.4e-3}"                # linear scaling from 6e-4 at global batch 32
WORKERS="${WORKERS:-21}"          # per GPU: 42 loaders + 2 ranks; 2 vCPU for NCCL/OS
PREFETCH="${PREFETCH:-1}"         # dense targets are large; one queued batch/worker
EVAL_BATCH="${EVAL_BATCH:-64}"    # per GPU
EVAL_WORKERS="${EVAL_WORKERS:-4}" # per GPU
EVAL_DECODE_WORKERS="${EVAL_DECODE_WORKERS:-18}"
CKPT_REPO="${CKPT_REPO:-BanVienCorp/LaneATT-Carla-checkpoints}"
CKPT_SUBDIR="${CKPT_SUBDIR:-carla-${VISION}}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EVAL_SUBSET="${EVAL_SUBSET:---eval-subset 500}"
# Resume a previous run in a NEW job: the container is fresh, so first pull last.pth
# from the model repo, then point --resume at it. Set RESUME_FROM_HUB=1 to do that.
RESUME_FROM_HUB="${RESUME_FROM_HUB:-0}"
RESUME="${RESUME:-}"
# For a quick pipeline smoke test first, set: SUBSET="--subset 64" and EPOCHS=1
SUBSET="${SUBSET:-}"
# ----------------------------------------------------------------------------

JOB_SCRIPT=$(cat <<EOF
set -eux

# Process-level parallelism owns the 46 vCPUs. Prevent OpenCV/BLAS from creating
# nested thread teams inside each of the 42 loader / 36 F1 decoder processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

pip install -q shapely opencv-python-headless huggingface_hub

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

# 4) DDP train on both H200s. The cold GT cache is built first with one sample per
#    worker task, then reused by every epoch. Checkpoints upload after every epoch.
torchrun --standalone --nproc_per_node=2 train.py --dataset carla \
    --data-root /workspace/ds/data/dataset \
    --label label_train.json \
    --eval-list label_val.json --eval-f1 --eval-every ${EVAL_EVERY} ${EVAL_SUBSET} \
    --eval-batch ${EVAL_BATCH} --eval-workers ${EVAL_WORKERS} \
    --eval-decode-workers ${EVAL_DECODE_WORKERS} \
    --vision ${VISION} --epochs ${EPOCHS} --batch ${BATCH} --lr ${LR} \
    --workers ${WORKERS} --prefetch ${PREFETCH} --warm-cache \
    --amp --amp-dtype bfloat16 --device cuda \
    --cache-dir /workspace/gt_cache \
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
    --env PYTHONUNBUFFERED=1 \
    --detach \
    "${IMAGE}" \
    bash -c "${JOB_SCRIPT}"
