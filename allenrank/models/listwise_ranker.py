from typing import Dict, Optional

from overrides import overrides
import torch

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
        embedded_text = self._text_field_embedder(query)
        mask = get_text_field_mask(query).long()

        embedded_documents = self._text_field_embedder(documents, num_wrapping_dims=1)
        documents_mask = get_text_field_mask(documents).long()

        if self._dropout:
            embedded_text = self._dropout(embedded_text)
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

        embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_documents.size(1), -1, -1) # [batch, num_documents, words, dim]
        mask = mask.unsqueeze(1).expand(-1, embedded_documents.size(1), -1)
        
        scores = self._relevance_matcher(embedded_text, embedded_documents, mask, documents_mask).squeeze(-1)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores, "probs": probs}
        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(query)
        if labels is not None:
            label_mask = (labels != -1)
            
            self._mrr(probs, labels, label_mask)
            self._ndcg(probs, labels, label_mask)
            
            probs = probs.view(-1)
            labels = labels.view(-1)
            label_mask = label_mask.view(-1)
            
            self._auc(probs, labels.ge(0.5).long(), label_mask)
            
            loss = self._loss(probs, labels)
            output_dict["loss"] = loss.masked_fill(~label_mask, 0).sum() / label_mask.sum()

        output_dict.update(kwargs)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {
            "auc": self._auc.get_metric(reset),
            "mrr": self._mrr.get_metric(reset),
            "ndcg": self._ndcg.get_metric(reset),
        }
        return metrics