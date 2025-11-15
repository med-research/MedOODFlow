#!/bin/bash
# sh scripts/ood/gmm/brats20_t1/brats20_t1_concat_test_gmm_feat.sh

SEED=0
MARK="5_feats"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
# Using the normalizing flow feature extractor as we only need to extract features of the backbone here
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

# Test with GMM on extracted features
# Using the normalizing flow test pipeline as the overall procedure is the same
# The network (Identity) is not used by the test pipeline here, as we directly load the extracted features
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/identity.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/gmm_feat.yml \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --dataset.feat_root "./results/brats20_t1_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/brats20_t1_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained False \
    --seed ${SEED} \
    --mark ${MARK}
