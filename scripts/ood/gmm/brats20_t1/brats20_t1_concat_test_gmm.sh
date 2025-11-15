#!/bin/bash
# sh scripts/ood/gmm/brats20_t1/brats20_t1_concat_test_gmm.sh

SEED=0
MARK="5_feats"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Test with GMM
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/resnet3d_18.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/med3d_preprocessor.yml \
    configs/postprocessors/gmm.yml \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
    --postprocessor.postprocessor_args.num_clusters_list 50 50 50 50 50 \
    --postprocessor.postprocessor_args.feature_type_list avg_pool_3d avg_pool_3d avg_pool_3d avg_pool_3d avg_pool_3d \
    --postprocessor.postprocessor_args.alpha_list 1 1 1 1 1 \
    --postprocessor.postprocessor_args.reduce_dim_list pca_50 pca_50 pca_50 pca_50 pca_50 \
    --seed ${SEED} \
    --mark ${MARK}
