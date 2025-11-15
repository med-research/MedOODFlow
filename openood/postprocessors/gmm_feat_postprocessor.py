from typing import Any

import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm

from .gmm_postprocessor import GMMPostprocessor
from .mds_ensemble_postprocessor import reduce_feature_dim, tensor2list


class GMMFeatPostprocessor(GMMPostprocessor):
    def __init__(self, config):
        super().__init__(config)

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        # Adapt get_GMM_stat to work with pre-extracted features
        # Features come in shape [batch, channels, 1, 1]
        self.feature_mean, self.feature_prec, \
            self.component_weight_list, self.transform_matrix_list = \
            self._get_GMM_stat_from_features(id_loader_dict['train'])

    @torch.no_grad()
    def _get_GMM_stat_from_features(self, train_loader):
        feature_all = []
        label_list = []

        # Collect features
        for batch in tqdm(train_loader, desc='Collect Features for GMM'):
            data = batch['data'].cuda()  # Pre-extracted features
            label = batch['label']
            # Features are in shape [batch, channels, 1, 1]
            feature_processed = data.flatten(1)
            feature_all.extend(tensor2list(feature_processed))
            label_list.extend(tensor2list(label))

        feature_all = np.array(feature_all)
        label_list = np.array(label_list)

        # Apply dimensionality reduction
        transform_matrix = reduce_feature_dim(feature_all, label_list,
                                              self.reduce_dim_list[0])
        feature_reduced = np.dot(feature_all, transform_matrix)

        # Fit GMM
        gm = GaussianMixture(
            n_components=self.num_clusters_list[0],
            random_state=0,
            covariance_type='tied',
        ).fit(feature_reduced)

        return ([torch.Tensor(gm.means_).cuda()
                 ], [torch.Tensor(gm.precisions_).cuda()
                     ], [torch.Tensor(gm.weights_).cuda()],
                [torch.Tensor(transform_matrix).cuda()])

    def postprocess(self, net: nn.Module, data: Any):
        # data is pre-extracted features [batch, channels, 1, 1]
        feature_list = data.flatten(1)
        feature_list = torch.mm(feature_list, self.transform_matrix_list[0])

        # Compute GMM score
        score = self._compute_gmm_score(feature_list)
        pred = torch.ones(data.shape[0], dtype=torch.long).cuda()
        return pred, score

    def _compute_gmm_score(self, features):
        prob_matrix = []
        for cluster_idx in range(len(self.feature_mean[0])):
            zero_f = features - self.feature_mean[0][cluster_idx]
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, self.feature_prec[0]),
                                       zero_f.t()).diag()
            prob_gau = torch.exp(term_gau)
            prob_matrix.append(prob_gau.view(-1, 1))

        prob_matrix = torch.cat(prob_matrix, 1)
        prob = torch.mm(prob_matrix, self.component_weight_list[0].view(-1, 1))
        return torch.log(prob.reshape(-1) + 1e-45)
