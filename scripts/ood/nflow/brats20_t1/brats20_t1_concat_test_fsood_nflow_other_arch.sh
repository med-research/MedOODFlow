#!/bin/bash
# sh scripts/ood/nflow/brats20_t1/brats20_t1_concat_test_fsood_nflow_other_arch.sh

SEED=0
MARK="5_feats"
NFLOW="nsf"  # other choices: glow, resflow
MARK2=""
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
# Since only extracting backbone features, it's independent of NFLOW choice
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_resnet3d_18_feat_concat.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    --pipeline.extract_nflow False \
    --network.pretrained False \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --seed ${SEED} \
    --mark ${MARK}

# evaluation
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/nflow_${NFLOW}_resnet3d_18_feat_concat.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/nflow.yml \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --dataset.feat_root "./results/brats20_t1_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/brats20_t1_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_${MARK}_${NFLOW}${MARK2}/s${SEED}/best_nflow.ckpt" None \
    --network.backbone.pretrained False \
    --nondeterministic_operators 'Warn' \
    --seed ${SEED} \
    --mark ${MARK}_${NFLOW}${MARK2}

# direct evaluation (without saving extracted features)
#python main.py \
#    --config configs/datasets/medood/brats20_t1.yml \
#    configs/datasets/medood/brats20_t1_fsood.yml \
#    configs/networks/nflow_${NFLOW}_resnet3d_18_feat_concat.yml \
#    configs/pipelines/test/test_nflow.yml \
#    configs/preprocessors/med3d_preprocessor.yml \
#    configs/postprocessors/nflow.yml \
#    --evaluator.ood_scheme fsood \
#    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
#    --network.pretrained True \
#    --network.checkpoint "./results/brats20_t1_nflow_nflow_e100_lr0.0001_${MARK}_${NFLOW}${MARK2}/s${SEED}/best_nflow.ckpt" None \
#    --network.backbone.pretrained False \
#    --network.backbone.encoder.pretrained True \
#    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
#    --nondeterministic_operators 'Warn' \
#    --seed ${SEED} \
#    --mark ${MARK}_${NFLOW}${MARK2}
