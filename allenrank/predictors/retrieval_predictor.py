from typing import List, Dict
from overrides import overrides

from collections import defaultdict
import torch

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance, DataLoader
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset

from tqdm.auto import tqdm
import logging
logger = logging.getLogger(__name__)


class BaseRetrievalPredictor(Predictor):
    def predict(self, query: str = None, document: str = None) -> JsonDict:
        if not (query or document):
            logger.warn('Both query and document are empty. Skipping.')
            return {}
        
        return self.predict_json({'query': query, 'document': document})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"query | document": "..."}`.
        Runs the underlying model.
        """
        return self._dataset_reader.text_to_instance(**json_dict)
    
    def _batch_instances(self, instances: List[Instance], batch_size=None):
        batch_size = batch_size or len(instances)
        dataset = AllennlpDataset(instances)
        dataset.index_with(self._model.vocab)
        return DataLoader(dataset, batch_size=batch_size)
    
    def _predict_batched_tensors(self, batch: Dict[str, torch.Tensor]):
        raise NotImplementedError()
    
    def predict_instance_dataset(self, instances: List[Instance], batch_size=None):
        output = defaultdict(list)
        for batch in tqdm(self._batch_instances(instances, batch_size)):
            pred = self._predict_batched_tensors(batch)
            for key, val in pred.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        output[f'{key}_{subkey}'].extend(subval)
                    continue
                output[key].extend(val)
        return dict(output)
    
    def predict_json_dataset(self, json_dicts: List[JsonDict], batch_size=None):
        instances = self._batch_json_to_instances(json_dicts)
        return self.predict_instance_dataset(instances, batch_size=batch_size)
    
    def predict_dataset(self, json_dicts: List[JsonDict], batch_size=None):
        return self.predict_json_dataset(json_dicts, batch_size=batch_size)