# Courtesy: https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/models/tk.py

from typing import Dict, Iterator, List
from overrides import overrides

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from allennlp.data import TextFieldTensors
from allennlp.modules.matrix_attention.cosine_matrix_attention import CosineMatrixAttention
from allennlp.modules.matrix_attention.dot_product_matrix_attention import *
from allennlp.modules import Seq2VecEncoder, GatedSum
from allennlp_models.rc.modules.seq2seq_encoders.stacked_self_attention import StackedSelfAttentionEncoder

from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.modules.kernels import KernelFunction, GaussianKernel

import torchsnooper

@KernelFunction.register('tk_gaussian')
class TKGaussianKernel(GaussianKernel):
    def __init__(
        self,
        n_kernels: int,
        window_size: List[int] = [20, 30, 50, 80, 100, 120, 150],
        max_windows: List[int] = None,
        max_sequence_length: int = 512,
        **kwargs
    ):
        super().__init__(n_kernels=n_kernels, **kwargs)

        self.window_size = window_size
        self.max_windows = max_windows or [math.ceil(max_sequence_length / w) for w in map(float, window_size)]

        self.kernel_weights = nn.ModuleList([nn.Linear(n_kernels, 1, bias=False) for w in window_size])
        self.nn_scaler = nn.ParameterList([nn.Parameter(torch.full([1], 0.01, requires_grad=True)).float() for w in window_size])

        window_scorer = []
        for w in self.max_windows:
            l =  nn.Linear(w, 1, bias=False)
            torch.nn.init.constant_(l.weight, 1/w)
            window_scorer.append(l)

        self.window_scorer = nn.ModuleList(window_scorer)
        self.window_merger = nn.Linear(len(self.window_size), 1, bias=False)

    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        document_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        document_mask: torch.Tensor = None
    ):
        mask = query_mask.unsqueeze(-1).float() @ document_mask.unsqueeze(-1).transpose(-1, -2).float()
        cosine_matrix = (self.cosine_module(query_embeddings, document_embeddings) * mask).unsqueeze(-1)
        
        raw_kernel_results = torch.exp(- torch.pow(cosine_matrix - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))
        kernel_results_masked = raw_kernel_results * mask.unsqueeze(-1)

        individual_window_scores = []
        for i, window in enumerate(self.window_size):

            kernel_results_masked = nn.functional.pad(kernel_results_masked, (0, 0, 0, window - kernel_results_masked.shape[-2] % window)) 

            scoring_windows = kernel_results_masked.unfold(dimension=-2, size=window, step=window)

            scoring_windows = scoring_windows.transpose(-1, -2)

            per_kernel_query = torch.sum(scoring_windows, -2)
            log_per_kernel_query = torch.log(torch.clamp(per_kernel_query, min=1e-10)) #* 
            log_per_kernel_query_masked = log_per_kernel_query * (per_kernel_query.sum(dim=-1) != 0).unsqueeze(-1).float()
            per_kernel = torch.sum(log_per_kernel_query_masked, 1) 

            window_scores = self.kernel_weights[i](per_kernel).squeeze(-1)

            window_scores_exp = torch.exp(window_scores * self.nn_scaler[i]) * (window_scores != 0).float()
            if window_scores_exp.shape[-1] > self.window_scorer[i].in_features:
                window_scores_exp = window_scores_exp[..., :self.window_scorer[i].in_features]
            if window_scores_exp.shape[-1] < self.window_scorer[i].in_features:
                window_scores_exp = nn.functional.pad(window_scores_exp, (0, self.window_scorer[i].in_features - window_scores_exp.shape[-1])) 
  
            window_scores_exp = window_scores_exp.sort(dim=-1, descending=True)[0]

            individual_window_scores.append(self.window_scorer[i](window_scores_exp))
        

        final_window_score = self.window_merger(torch.cat(individual_window_scores, dim=1))
        return final_window_score

@RelevanceMatcher.register('tk')
class TransformerKernel(RelevanceMatcher):
    '''
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions

    Default values from: https://github.com/sebastian-hofstaetter/transformer-kernel-ranking/blob/master/matchmaker/configs/models/model-config.yaml 
    '''
    def __init__(
        self,
        embedding_dim: int,
        att_layer: int = 2,
        att_proj_dim: int = 32,
        att_ff_dim: int = 100,
        num_heads: int = 8,
        
        n_kernels: int = 11,
        window_size: List[int] = [20, 30, 50, 80, 100, 120, 150],
        max_windows: List[int] = None,
        max_sequence_length: int = 512,
    ):
        super().__init__()

        self.stacked_att = StackedSelfAttentionEncoder(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            projection_dim=att_proj_dim,
            feedforward_hidden_dim=att_ff_dim,
            num_layers=att_layer,
            num_attention_heads=num_heads,
            dropout_prob = 0,
            residual_dropout_prob = 0,
            attention_dropout_prob = 0
        )
        self.mixer = GatedSum(self.stacked_att.get_output_dim())
        self.kernel = TKGaussianKernel(n_kernels, window_size, max_windows, max_sequence_length)


    def forward(
        self, 
        query_embeddings: torch.Tensor, 
        document_embeddings: torch.Tensor,
        query_mask: torch.Tensor, 
        document_mask: torch.Tensor,
    ) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings_context = self.stacked_att(query_embeddings, query_mask.bool())
        document_embeddings_context = self.stacked_att(document_embeddings, document_mask.bool())

        # query_embeddings = query_embeddings * query_mask.unsqueeze(-1)
        # document_embeddings = document_embeddings * document_mask.unsqueeze(-1)

        query_embeddings = self.mixer(query_embeddings, query_embeddings_context) * query_mask.unsqueeze(-1)
        document_embeddings = self.mixer(document_embeddings, document_embeddings_context) * document_mask.unsqueeze(-1)
       
        score = self.kernel(
            query_embeddings,
            document_embeddings,
            query_mask,
            document_mask
        )
        return score.squeeze(-1)