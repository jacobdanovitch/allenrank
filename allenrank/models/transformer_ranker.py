from typing import Dict, Optional

from overrides import overrides
import torch
from torch import nn
from torch.nn import functional as F

from allennlp.data import TextFieldTensors
from allennlp.models.model import Model
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.seq2vec_encoders.cls_pooler import ClsPooler

from allenrank.models import PairwiseDocumentRanker

import torchsnooper


@Model.register("transformer_ranker")
class TransformerDocumentRanker(PairwiseDocumentRanker):
    _document_input_key: str = 'text'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cls_pool = ClsPooler()
        self._scorer = nn.Linear(self._text_field_embedder.get_output_dim(), 1)

    def forward(  # type: ignore
        self,
        text: TextFieldTensors, # batch * words
        label: torch.IntTensor = None, # batch * 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        embedded = self._text_field_embedder(text)
        mask = get_text_field_mask(text).long()

        if self._dropout:
            embedded = self._dropout(embedded)
        
        encoded = self._cls_pool(embedded, mask)
        scores = self._scorer(encoded)

        if scores.dim() > 1:
            scores = scores.squeeze(-1)
        probs = torch.sigmoid(scores)

        output_dict = {"logits": scores, "scores": probs}
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