from typing import Dict, List, Union, Tuple
from overrides import overrides

import numpy as np
import pandas as pd

from allennlp.data.fields import ArrayField, MetadataField, ListField
from allennlp.data.instance import Instance

from allenrank.dataset_readers.ir_reader import IRDatasetReader

class ListwiseRankingReader(IRDatasetReader):
    @overrides
    def text_to_instance(
        self,
        query: Union[str, Tuple], 
        documents: List[str],
        labels: Union[str, float] = None,
        **kwargs
    ) -> Instance:  # type: ignore
        documents = list(filter(None, documents))
        
        if labels:
            assert all(l >= 0 for l in labels)
            assert all((l == 0) for l in labels[len(documents):])
            labels = labels[:len(documents)]
            
        
        query_field = self._make_textfield(query)
        documents_field = ListField([self._make_textfield(o) for o in documents])
        
        fields = { 'query': query_field, 'documents': documents_field, 'metadata': MetadataField(kwargs) }

        if labels:
            labels = list(map(float, filter(lambda x: not pd.isnull(x), labels)))            
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        fields = {k: v for (k, v) in fields.items() if v is not None}
        return Instance(fields)