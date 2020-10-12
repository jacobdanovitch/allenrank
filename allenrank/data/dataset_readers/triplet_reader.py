


from typing import Dict, List, Union, Tuple
from overrides import overrides

import numpy as np
import pandas as pd

from allennlp.data.fields import ArrayField, MetadataField
from allennlp.data.instance import Instance

from allenrank.data.dataset_readers.pairwise_reader import PairwiseRankingReader


class TripletRankingReader(PairwiseRankingReader):
    @overrides
    def text_to_instance(
        self,
        query: Union[str, Tuple],
        positive_document: str,
        negative_document: str = None,
        **kwargs
    ) -> Instance:  # type: ignore
        
        query_field = self._make_textfield(query)
        positive_document_field = self._make_textfield(positive_document)
        
        fields = { 
            'query': query_field, 
            'positive_document': positive_document_field, 
            'metadata': MetadataField(kwargs) 
        }
        if negative_document is not None:
            fields['negative_document'] = self._make_textfield(negative_document)
        
        return Instance(fields)