from typing import Dict, Optional
from overrides import overrides

import torch
import torch.nn as nn
from torch.nn import functional as F

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder, TimeDistributed
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Auc, F1Measure, PearsonCorrelation

from allenrank.models.document_ranker import DocumentRanker
from allenrank.modules.relevance.base import RelevanceMatcher
from allenrank.training.metrics.multilabel_f1 import MultiLabelF1Measure
from allenrank.training.metrics import MRR, NDCG

import torchsnooper


@Model.register("listwise_ranker")
class ListwiseDocumentRanker(DocumentRanker):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        relevance_matcher: RelevanceMatcher,
        **kwargs
    ) -> None:
        super().__init__(vocab, text_field_embedder, TimeDistributed(relevance_matcher), **kwargs)

        self._loss = torch.nn.MSELoss(reduction='none')

        self._mrr = MRR(padding_value=-1)
        self._ndcg = NDCG(padding_value=-1)

    @overrides
    def forward(  # type: ignore
        self,
        query: TextFieldTensors, # batch * words
        documents: TextFieldTensors, # batch * num_documents * words
        labels: torch.IntTensor = None, # batch * num_documents,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        embedded_query, query_mask = self._embed_and_mask(query)
        embedded_documents, documents_mask = self._embed_and_mask(documents)

        if self._dropout:
            embedded_query = self._dropout(embedded_query)
            embedded_documents = self._dropout(embedded_documents)

        """
        This isn't exactly a 'hack', but it's definitely not the most efficient way to do it.
        Our matcher expects a single (query, document) pair, but we have (query, [d_0, ..., d_n]).
        To get around this, we expand the query embeddings to create these pairs, and then
        flatten both into the 3D tensor [batch*num_documents, words, dim] expected by the matcher. 
        The expansion does this:

        [
            (q_0, [d_{0,0}, ..., d_{0,n}]), 
            (q_1, [d_{1,0}, ..., d_{1,n}])
        ]
        =>
        [
            [ (q_0, d_{0,0}), ..., (q_0, d_{0,n}) ],
            [ (q_1, d_{1,0}), ..., (q_1, d_{1,n}) ]
        ]

        Which we then flatten along the batch dimension. It would likely be more efficient
        to rewrite the matrix multiplications in the relevance matchers, but this is a more general solution.
        """

        embedded_query = embedded_query.unsqueeze(1).expand(-1, embedded_documents.size(1), -1, -1) # [batch, num_documents, words, dim]
        query_mask = query_mask.unsqueeze(1).expand(-1, embedded_documents.size(1), -1)
        
        scores = self._relevance_matcher(embedded_query, embedded_documents, query_mask, documents_mask).squeeze(-1)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores, "scores": probs}
        if labels is not None:
            label_mask = (labels != -1)
            output_dict["loss"] = self._letor(probs, labels, label_mask)
            
            self._mrr(probs, labels, label_mask)
            self._ndcg(probs, labels, label_mask)
            
            probs = probs.view(-1)
            labels = labels.view(-1)
            label_mask = label_mask.view(-1)
            
            self._auc(probs, labels.ge(0.5).long(), label_mask)

            multi_class = torch.stack([1-probs, probs], dim=-1)
            self._f1(multi_class, labels.ge(0.5).long(), label_mask)

        output_dict.update(kwargs)
        return output_dict

    def embed(self, query: TextFieldTensors = None, document: TextFieldTensors = None, **kwargs):
        prefix = 'query' if query else 'document'
        matcher = self._relevance_matcher._module
        encoder = getattr(matcher, '_{}_seq2vec_encoder'.format(prefix))

        embedded, mask = self._embed_and_mask(query or document)
        encoded = encoder(embedded, mask)

        return {'encoder_embeddings': encoded, **kwargs}

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "auc": self._auc.get_metric(reset),
            "mrr": self._mrr.get_metric(reset),
            "ndcg": self._ndcg.get_metric(reset),
            **self._f1.get_metric(reset)
        }
        return metrics