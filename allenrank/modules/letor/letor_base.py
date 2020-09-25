import torch
from torch import nn

from allennlp.common.registrable import Registrable

class LearningToRankLayer(Registrable, nn.Module):
    def forward(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        label_mask: torch.Tensor = None
    ):
        raise NotImplementedError()