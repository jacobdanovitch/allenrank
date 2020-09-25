from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Auc, F1Measure, FBetaMeasure, PearsonCorrelation

from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.training.metrics.multilabel_f1 import MultiLabelF1Measure
from allenrank.training.metrics import NDCG, MRR

import torchsnooper

class DocumentRanker(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        relevance_matcher: RelevanceMatcher,
        dropout: float = 0.,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._relevance_matcher = relevance_matcher

        self._dropout = torch.nn.Dropout(dropout)

        self._auc = Auc()
        self._f1 = F1Measure(positive_label=1)
        
        initializer(self)

    def _embed_and_mask(self, source_tokens: TextFieldTensors):
        if source_tokens is None:
            return None, None
        
        token_ids = util.get_token_ids_from_text_field_tensors(source_tokens)
        source_mask = util.get_text_field_mask(source_tokens, num_wrapping_dims=token_ids.dim()-2)
        embedded_input = self._text_field_embedder(source_tokens, num_wrapping_dims=token_ids.dim()-2)

        return embedded_input, source_mask

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return output_dict