from typing import Dict, Iterator, List
from overrides import overrides

import torch
import torch.nn as nn
from torch.nn import functional as F

from allennlp.modules import Seq2VecEncoder
from allennlp.data import TextFieldTensors

from allenrank.modules.relevance.base import RelevanceMatcher

import torchsnooper

@RelevanceMatcher.register('cosine')
class CosineSimilarityMatcher(RelevanceMatcher):
    def __init__(
        self,
        query_seq2vec_encoder: Seq2VecEncoder,
        document_seq2vec_encoder: Seq2VecEncoder = None,
    ):
        super().__init__()#input_dim=query_seq2vec_encoder.get_input_dim())
        self._query_seq2vec_encoder = query_seq2vec_encoder
        self._document_seq2vec_encoder = document_seq2vec_encoder or query_seq2vec_encoder

    @overrides
    def forward(
        self, 
        query_embeddings: TextFieldTensors, 
        document_embeddings: TextFieldTensors,
        query_mask: torch.Tensor = None,
        document_mask: torch.Tensor = None,
    ) -> torch.Tensor:

        query_encoded = self._query_seq2vec_encoder(query_embeddings, query_mask)
        document_encoded = self._document_seq2vec_encoder(document_embeddings, document_mask)

        score = F.cosine_similarity(query_encoded, document_encoded)
        return score