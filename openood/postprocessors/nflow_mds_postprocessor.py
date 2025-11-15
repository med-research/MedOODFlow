from typing import Any

import numpy as np
import sklearn
import torch
from normflows.distributions import DiagGaussian
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor


class NormalizingFlowMDSPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(NormalizingFlowMDSPostprocessor, self).__init__(config)
        self.mean = None
        self.precision = None
        self.setup_flag = False
        self.num_classes = config.dataset.num_classes
        self.per_class = config.postprocessor.get('per_class', None)

    def setup(self, net, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label'].cuda()
                    if data.shape[-1] > 1 and data.shape[1] in {1, 3}:
                        data = torch.Tensor(data)
                        output, feats = net['backbone'](data,
                                                        return_feature=True)
                        score = torch.softmax(output, dim=1)
                        _, pred = torch.max(score, dim=1)
                        if 'feat_agg' in net:
                            feats = net['feat_agg'](feats)
                    elif data.shape[-1] == 1 and data.shape[-2] == 1:
                        feats = data.flatten(1)
                        if 'feat_agg' in net:
                            feats = net['feat_agg'](feats)
                        # dummy predictions
                        pred = torch.ones((feats.shape[0], )).cuda()
                    else:
                        raise ValueError('Unsupported input type!')
                    feats = net['nflow'].inverse(feats)
                    all_feats.append(feats)
                    all_labels.append(labels)
                    all_preds.append(pred)

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            if self.per_class is not None:
                # compute class means
                class_mean = []
                centered_data = []
                for c in range(self.num_classes):
                    class_samples = all_feats[all_labels.eq(c)].data
                    class_mean.append(class_samples.mean(0))
                    centered_data.append(class_samples -
                                         class_mean[c].view(1, -1))
                self.mean = torch.stack(class_mean)
                centered_data = torch.cat(centered_data)
            else:
                # compute global mean
                self.mean = torch.mean(all_feats, dim=0)
                # center the data using the global mean
                centered_data = all_feats - self.mean.view(1, -1)

            # compute precision matrix (inverse of covariance matrix)
            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            # inverse of covariance
            self.precision = torch.from_numpy(group_lasso.precision_) \
                .float().cuda()
            self.setup_flag = True
        else:
            pass

    def compute_mds_and_log_prob(self, nflow_net, feats):
        if not hasattr(nflow_net, 'q0') or \
           not isinstance(nflow_net.q0, DiagGaussian):
            raise ValueError('The Normalizing Flow network does not'
                             ' contain a valid q0 distribution.')

        z, log_det = nflow_net.inverse_and_log_det(feats)
        log_prob = log_det + nflow_net.q0.log_prob(z)

        mean = self.mean
        precision = self.precision

        if self.mean is None or self.precision is None:
            # Get the mean and covariance from the Normalizing Flow network
            mean = nflow_net.q0.loc.squeeze(0)
            # Variances are exp(log_scale)^2
            variances = torch.exp(nflow_net.q0.log_scale.squeeze(0))**2
            # Covariance matrix is diagonal with variances
            cov_matrix = torch.diag(variances)
            # Calculate precision matrix (inverse of covariance)
            try:
                precision = torch.inverse(cov_matrix)
            except RuntimeError:
                # Add small regularization if matrix is singular
                reg_cov = cov_matrix + torch.eye(cov_matrix.shape[0],
                                                 dtype=torch.float) * 1e-6
                precision = torch.inverse(reg_cov)

        if self.per_class is None or \
           self.mean is None or self.precision is None:
            # Compute the Mahalanobis distance
            centered_z = z - mean.view(1, -1)
            mds = torch.matmul(torch.matmul(centered_z, precision),
                               centered_z.t()).diag()
        else:
            class_mds = torch.zeros((z.shape[0], self.num_classes)).cuda()
            for c in range(self.num_classes):
                centered_z = z - mean[c].view(1, -1).cuda()
                class_mds[:, c] = torch.matmul(
                    torch.matmul(centered_z, precision.cuda()),
                    centered_z.t()).diag()
            if self.per_class == 'avg':
                mds = torch.mean(class_mds, dim=1)
            elif self.per_class == 'max':
                mds = torch.max(class_mds, dim=1)[0]
            elif self.per_class == 'min':
                mds = torch.min(class_mds, dim=1)[0]
            else:
                raise ValueError(f'Unsupported per_class: {self.per_class}')

        return mds, log_prob

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        # images input
        if data.shape[-1] > 1 and data.shape[1] in {1, 3}:
            data = torch.Tensor(data)  # fix the issue when data is MetaTensor
            output, feats = net['backbone'](data, return_feature=True)
            score = torch.softmax(output, dim=1)
            _, pred = torch.max(score, dim=1)
            if 'feat_agg' in net:
                feats = net['feat_agg'](feats)
            mds, log_prob = self.compute_mds_and_log_prob(net['nflow'], feats)
            conf = (log_prob - mds).detach().cpu()
        # feature input
        elif data.shape[-1] == 1 and data.shape[-2] == 1:
            feats = data.flatten(1)
            if 'feat_agg' in net:
                feats = net['feat_agg'](feats)
            mds, log_prob = self.compute_mds_and_log_prob(net['nflow'], feats)
            conf = (log_prob - mds).detach().cpu()
            pred = torch.ones_like(conf)  # dummy predictions
        else:
            raise ValueError('Unsupported input type!')

        return pred, conf
