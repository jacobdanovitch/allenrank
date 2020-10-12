from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn
from torch.nn import functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, F1Measure

from allenrank.models import PairwiseDocumentRanker
from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.training.metrics import MRR, NDCG

import torchsnooper


@Model.register("triplet_ranker")
class TripletDocumentRanker(PairwiseDocumentRanker):
    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self._margin = margin

        self._acc = BooleanAccuracy()
        self._f1 = F1Measure(positive_label=1)
        self._mrr = MRR(padding_value=-1)
        self._ndcg = NDCG(padding_value=-1)

    # @torchsnooper.snoop()
    @overrides
    def forward(  # type: ignore
        self,
        query: TextFieldTensors,
        positive_document: TextFieldTensors,
        negative_document: TextFieldTensors = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        embedded_query = self._text_field_embedder(query)
        query_mask = util.get_text_field_mask(query).long()

        embedded_positive_document = self._text_field_embedder(positive_document)
        positive_document_mask = util.get_text_field_mask(positive_document).long()

        if self._dropout:
            embedded_query = self._dropout(embedded_query)
            embedded_positive_document = self._dropout(embedded_positive_document)
        
        pos_scores = self._relevance_matcher(embedded_query, embedded_positive_document, query_mask, positive_document_mask)
        pos_probs = torch.sigmoid(pos_scores)

        if negative_document is not None:
            output_dict = {"pos_scores": pos_probs}

            embedded_negative_document = self._text_field_embedder(negative_document)
            negative_document_mask = util.get_text_field_mask(negative_document).long()

            if self._dropout:
                embedded_negative_document = self._dropout(embedded_negative_document)
            
            neg_scores = self._relevance_matcher(embedded_query, embedded_negative_document, query_mask, negative_document_mask)
            label = torch.ones(embedded_query.size(0)).to(embedded_query.device)

            output_dict["loss"] = F.margin_ranking_loss(pos_scores, neg_scores, label, margin=self._margin)

            preds = (pos_scores - neg_scores).gt(self._margin).long()
            probs = torch.cat([pos_scores.view(-1, 1), neg_scores.view(-1, 1)], dim=-1) # B x 2
            self._acc(preds, label.long())
            # self._f1(probs, label)

            multi_label = torch.stack([label, torch.zeros_like(label)], dim=-1).long() # B x 2
            self._mrr(probs, multi_label)
            self._ndcg(probs, multi_label.float())

            probs = probs.view(-1)
            multi_class = torch.stack([1-probs, probs], dim=-1)
            multi_label = multi_label.view(-1)
            # self._auc(probs, multi_label)
            self._f1(multi_class, multi_label)
        else:
            output_dict = {"scores": pos_probs}

        output_dict.update(kwargs)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "acc": self._acc.get_metric(reset),
            # "auc": self._auc.get_metric(reset),
            "mrr": self._mrr.get_metric(reset),
            "ndcg": self._ndcg.get_metric(reset),
            **self._f1.get_metric(reset)
        }
        return metrics