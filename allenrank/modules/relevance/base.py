import torch
from torch import nn

from allennlp.common.registrable import Registrable
from allennlp.data import TextFieldTensors

class RelevanceMatcher(Registrable, nn.Module):
    def __init__(
        self,
        input_dim: int = None
    ):
        super().__init__()
        self._input_dim = input_dim
        if input_dim:
            self.dense = nn.Linear(input_dim, 1, bias=False)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return 1

    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        candidates_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        candidates_mask: torch.Tensor = None
    ):
        raise NotImplementedError()