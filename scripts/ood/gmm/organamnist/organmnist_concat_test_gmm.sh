#!/bin/bash
# sh scripts/ood/gmm/organmnist/organmnist_concat_test_gmm.sh

SEED=0
MARK="5_feats"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Test with GMM
python main.py \
    --config configs/datasets/medmnist/organamnist.yml \
    configs/datasets/medmnist/organamnist_ood.yml \
    configs/networks/resnet18_28x28.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gmm.yml \
    --network.pretrained True \
    --network.checkpoint "./results/organamnist_resnet18_28x28/s${SEED}/resnet18_28_1.pth" \
    --network.checkpoint_key "net" \
    --postprocessor.postprocessor_args.num_clusters_list 50 50 50 50 50 \
    --postprocessor.postprocessor_args.feature_type_list avg_pool_2d avg_pool_2d avg_pool_2d avg_pool_2d avg_pool_2d \
    --postprocessor.postprocessor_args.alpha_list 1 1 1 1 1 \
    --postprocessor.postprocessor_args.reduce_dim_list pca_50 pca_50 pca_50 pca_50 pca_50 \
    --seed ${SEED} \
    --mark ${MARK}
