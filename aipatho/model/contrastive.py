#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision

# class AddProjection(nn.Module):
#     def __init__(self, config, model=None, mlp_dim=512):
#         super(AddProjection, self).__init__()
#
#         embedding_size = config.embedding_size
#         self.backbone = self.default(model, torchvision.models.resnet18(pretrained=False, num_classes=config.embedding_size))
#         mlp_dim = self.default(mlp_dim, self.backbone.fc.in_features)
#         print('Dim MLP input:', mlp_dim)
#         self.backbone.fc = nn.Identity()
#
#         # add mlp projection head
#         self.projection = nn.Sequential(
#             nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
#             # nn.BatchNorm1d(mlp_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=mlp_dim, out_features=embedding_size),
#             # nn.BatchNorm1d(embedding_size),
#         )
#
#     def forward(self, x, return_embedding=False):
#         embedding = self.backbone(x)
#         if return_embedding:
#             return embedding
#         return self.projection(embedding)
#
#     @staticmethod
#     def default(val, def_val):
#         return def_val if val is None else val


class SimCLR(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int = 512):
        super().__init__()

        # self.model = AddProjection(config, model=model, mlp_dim=feat_dim)

        self.backbone = backbone

        mlp_dim = self.backbone.fc.in_features
        embedding_size = self.backbone.fc.out_features

        # Remove FC layer
        self.backbone.fc = nn.Identity()
        # Add MLP projection head
        self.projection = nn.Sequential(
            nn.Linear(in_features=mlp_dim, out_features=mlp_dim),
            # nn.BatchNorm1d(mlp_dim),
            nn.ReLU(),
            nn.Linear(in_features=mlp_dim, out_features=embedding_size),
            # nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.projection(x)
        return x

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x
