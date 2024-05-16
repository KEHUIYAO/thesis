import numpy as np
import pandas as pd
import torch
import itertools
from torch import nn


class MeanModel(nn.Module):
    def forward(self,
                x,
                mask,
                **kwargs):

        # x: [batch, steps, nodes, channels]
        B = x.shape[0]
        K = x.shape[2]
        C = x.shape[3]
        for b in range(B):
            for k in range(K):
                for c in range(C):
                    x[b, :, k, c] = torch.mean(x[b, mask[b, :, k, c].bool(), k, c])


        x[torch.isnan(x)] = 0

        return x

    @staticmethod
    def add_model_specific_args(parser):
        return parser