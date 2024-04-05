#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
    """
    def __init__(self, temperature: float = 0.5, device: str = 'cuda'):
        super().__init__()
        # self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        # self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()

    @staticmethod
    def calc_similarity_batch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

    def forward(self, proj_1: torch.Tensor, proj_2: torch.Tensor) -> torch.Tensor:
        """
        proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
        where corresponding indices are pairs
        z_i, z_j in the SimCLR paper

        :param proj_1:  Original embedding [batch, embedding_dim]
        :param proj_2:  Augmented embedding [batch, embedding_dim]
        """
        batch_size = proj_1.shape[0]
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(self.device)
        z_i = F.normalize(proj_1, p=2, dim=1)
        z_j = F.normalize(proj_2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j).to(self.device)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)

        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, device: str = 'cuda'):
        super(InfoNCELoss, self).__init__()

        self._temperature = temperature
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logit, _ = self.info_nce(x)

        # print("-------- InfoNCELoss --------")
        # print(logit.shape)
        # print(logit)
        # print(y.shape)
        # print(y)

        return self.criterion(logit, y)

    def info_nce(self, x: torch.Tensor):
        print("--- Calculating info_nce.................")
        batch_size = x.shape[0]
        n_views = 1

        # labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
        labels = torch.arange(batch_size)
        print(labels)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        print(labels)
        labels = labels.to(x.device)

        # バッチ
        features = F.normalize(x, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(x.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # Select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logit = torch.cat([positives, negatives], dim=1)
        logit = logit / self._temperature

        # FIXME: Is this correct...?
        labels = torch.zeros(logit.shape[0], dtype=torch.long).to(self.device)

        return logit, labels



class SupConLoss(nn.Module):
    """
    うまくいかなかったので使ってない by Nonaka @2024/Mar/*

    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """

    def __init__(self,
                 temperature,
                 contrast_mode='all',
                 base_temperature=0.07,
                 device: str = None
                 ):
        super(SupConLoss, self).__init__()

        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

        self.__device = device

    @property
    def device(self):
        return self.__device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # print(batch_size)
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # print(exp_logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(logits.shape)
        # print(exp_logits.sum(1, keepdim=True).shape)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # print(loss)
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
