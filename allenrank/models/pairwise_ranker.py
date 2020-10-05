from typing import Dict, Optional

from overrides import overrides
import torch
from torch.nn import functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Auc, F1Measure, FBetaMeasure, PearsonCorrelation

from allenrank.models.document_ranker import DocumentRanker
from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.training.metrics.multilabel_f1 import MultiLabelF1Measure
from allenrank.training.metrics import NDCG, MRR

import torchsnooper


@Model.register("pairwise_ranker")
class PairwiseDocumentRanker(DocumentRanker):
    def forward(  # type: ignore
        self,
        query: TextFieldTensors, # batch * words
        document: TextFieldTensors, # batch * words
        label: torch.IntTensor = None, # batch * 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        embedded_query = self._text_field_embedder(query)
        query_mask = get_text_field_mask(query).long()

        embedded_document = self._text_field_embedder(document)
        document_mask = get_text_field_mask(document).long()

        if self._dropout:
            embedded_query = self._dropout(embedded_query)
            embedded_document = self._dropout(embedded_document)
        
        scores = self._relevance_matcher(embedded_query, embedded_document, query_mask, document_mask)
        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores, "scores": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(query)
        if label is not None:
            probs = probs.view(-1)
            label = label.view(-1)
            
            self._auc(probs, label.ge(0.5).long())
            output_dict["loss"] = F.mse_loss(probs, label)

        output_dict.update(kwargs)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "auc": self._auc.get_metric(reset),
        }
        return metrics