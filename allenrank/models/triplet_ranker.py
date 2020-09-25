from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn
from torch.nn import functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, util

from allenrank.models.listwise_ranker import ListwiseDocumentRanker
from allenrank.modules.relevance.base import RelevanceMatcher

import torchsnooper


@Model.register("triplet_ranker")
class TripletDocumentRanker(ListwiseDocumentRanker):
    @overrides
    def forward(  # type: ignore
        self,
        query: TextFieldTensors, # batch * words
        positive_document: TextFieldTensors,
        negative_documents: TextFieldTensors, # batch * num_documents * words
        labels: torch.IntTensor = None, # batch * num_documents,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()