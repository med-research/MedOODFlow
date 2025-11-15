#!/bin/bash
# sh scripts/basics/brats20_t1/train_brats20_t1_swin_vit3d.sh

SEED=0
MARK="default"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# train
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/networks/swin_vit3d.yml \
    configs/pipelines/train/train_med3d.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --loss.name 'focal' \
    --loss.weight 2.45 0.63 \
    --scheduler.name 'warmup+cosine' \
    --scheduler.warmup_epochs 5 \
    --optimizer.num_epochs 200 \
    --seed ${SEED} \
    --mark ${MARK}
