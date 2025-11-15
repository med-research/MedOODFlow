#!/bin/bash
# sh scripts/ood/gmm/organmnist/organmnist_concat_test_gmm_feat.sh

SEED=0
MARK="5_feats"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# feature extraction
# Using the normalizing flow feature extractor as we only need to extract features of the backbone here
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/nflow_resnet18_28x28_feat_concat.yml \
    configs/pipelines/test/feat_extract_nflow.yml \
    configs/preprocessors/base_preprocessor.yml \
    --pipeline.extract_nflow False \
    --network.pretrained False \
    --network.backbone.pretrained False \
    --network.backbone.encoder.pretrained True \
    --network.backbone.encoder.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.backbone.encoder.checkpoint_key "net" \
    --seed ${SEED} \
    --mark ${MARK}

# Test with GMM on extracted features
# Using the normalizing flow test pipeline as the overall procedure is the same
# The network (Identity) is not used by the test pipeline here, as we directly load the extracted features
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/identity.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/gmm_feat.yml \
    --dataset.feat_root "./results/organamnist_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/organamnist_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained False \
    --seed ${SEED} \
    --mark ${MARK}
