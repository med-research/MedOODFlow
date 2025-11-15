#!/bin/bash
# sh scripts/ood/vae/brats20_t1/brats20_t1_concat_test_fsood_vae.sh

SEED=0
MARK="5_feats"

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

# evaluation
# Using the normalizing flow test pipeline as the overall procedure is the same
python main.py \
    --config configs/datasets/medood/brats20_t1.yml \
    configs/datasets/medood/brats20_t1_fsood.yml \
    configs/networks/vae_resnet3d_18_feat_concat.yml \
    configs/pipelines/test/test_nflow.yml \
    configs/postprocessors/vae.yml \
    --evaluator.ood_scheme fsood \
    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
    --dataset.feat_root "./results/brats20_t1_feat_concat_feat_extract_nflow_${MARK}/s${SEED}" \
    --ood_dataset.feat_root "./results/brats20_t1_nflow_feat_extract_nflow_${MARK}/s${SEED}" \
    --network.pretrained True \
    --network.checkpoint "./results/brats20_t1_vae_vae_e100_lr0.0001_${MARK}/s${SEED}/best_vae.ckpt" None \
    --network.backbone.pretrained False \
    --seed ${SEED} \
    --mark ${MARK}

# direct evaluation (without saving extracted features)
#python main.py \
#    --config configs/datasets/medood/brats20_t1.yml \
#    configs/datasets/medood/brats20_t1_fsood.yml \
#    configs/networks/vae_resnet3d_18_feat_concat.yml \
#    configs/pipelines/test/test_nflow.yml \
#    configs/preprocessors/med3d_preprocessor.yml \
#    configs/postprocessors/vae.yml \
#    --evaluator.ood_scheme fsood \
#    --evaluator.ood_splits transformation_shift population_shift modality_shift diagnostic_shift organ_shift \
#    --network.pretrained True \
#    --network.checkpoint "./results/brats20_t1_vae_vae_e100_lr0.0001_${MARK}/s${SEED}/best_vae.ckpt" None \
#    --network.backbone.pretrained False \
#    --network.backbone.encoder.pretrained True \
#    --network.backbone.encoder.checkpoint "./results/brats20_t1_resnet3d_18_med3d_e100_lr0.0001_default/s${SEED}/best.ckpt" \
#    --seed ${SEED} \
#    --mark ${MARK}
