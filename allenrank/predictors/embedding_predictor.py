from typing import List, Dict
from overrides import overrides

import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from allenrank.predictors.retrieval_predictor import BaseRetrievalPredictor

import logging
logger = logging.getLogger(__name__)


@Predictor.register("embedding_predictor", exist_ok=True)
class EmbeddingPredictor(BaseRetrievalPredictor):
    @overrides
    def _predict_batched_tensors(self, batch: Dict[str, torch.Tensor]):
        outputs = {'metadata': batch.get('metadata', None)}
        if 'query' in batch:
            outputs['query'] = sanitize(self._model._encode(batch['query'], return_dict=True))
            
        if 'document' in batch:
            outputs['document'] = sanitize(self._model._encode(batch['document'], return_dict=True))
        return outputs
    
    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        batch = next(iter(self._batch_instances(instances)))
        return self._predict_batched_tensors(batch)