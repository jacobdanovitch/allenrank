from typing import Dict, List, Union, Tuple
from overrides import overrides

import numpy as np
import pandas as pd

from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance

from allenrank.data.dataset_readers.ir_reader import IRDatasetReader


class PairwiseRankingReader(IRDatasetReader):
    @overrides
    def text_to_instance(
        self,
        query: Union[str, Tuple] = None,
        document: str = None,
        label: Union[str, float] = None,
        **kwargs
    ) -> Instance:  # type: ignore
        
        query_field = self._make_textfield(query)
        document_field = self._make_textfield(document)
        
        fields = { 'query': query_field, 'document': document_field, 'metadata': MetadataField(kwargs) }

        if not pd.isnull(label):
            label = np.array([label]).astype(float)
            fields['label'] = ArrayField(label, padding_value=-1)
        
        fields = {k: v for (k, v) in fields.items() if v is not None}
        return Instance(fields)


class TransformerRankingReader(IRDatasetReader):
    @overrides
    def text_to_instance(
        self,
        query: Union[str, Tuple] = None,
        document: str = None,
        label: Union[str, float] = None,
        **kwargs
    ) -> Instance:  # type: ignore
        
        text_field = self._make_textfield((query, document))
        
        fields = { 'text': text_field, 'metadata': MetadataField(kwargs) }

        if not pd.isnull(label):
            label = np.array([label]).astype(float)
            fields['label'] = ArrayField(label, padding_value=-1)
        
        fields = {k: v for (k, v) in fields.items() if v is not None}
        return Instance(fields)