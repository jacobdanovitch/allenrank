from typing import Union, List, Dict, Optional, Any
from collections import OrderedDict
from allennlp.common import Registrable

class NegativeSampler(Registrable):
    def retrieve(self, document: Union[str, List[str]], top_k: int = 10) -> Union[List[str], List[List[str]]]:
        raise NotImplementedError()
        
    def write_documents(self, documents: List[str]) -> None:
        raise NotImplementedError()

    def _build_document_mapping(self, documents: List[str]) -> Dict[str, int]:
        index = OrderedDict(map(reversed, enumerate(sorted(documents))))
        assert len(index.values()) == len(set(index.values()))
        self._index = index
        return index

    def _document_to_dict(self, document: Union[str, Dict[str, str]], doc_id: int = None, key: str = 'text') -> Dict[str, str]:
        if isinstance(document, str):
            d = {key: document}
            if doc_id is not None:
                d['id'] = doc_id
            return d
        
        # converts something like a pandas row to true dict
        return dict(d)
    
    def documents_to_dicts(self, documents: List[str], key: str = 'text') -> List[Dict[str, str]]:
        return [self._document_to_dict(d) for d in documents]


class OfflineSamplerMixin:
    def __init__(
        self,
        sampler: NegativeSampler,
        top_k: int = 10,
        max_documents_to_index: int = None
    ):
        self._sampler = sampler
        self._top_k = top_k
        self._max_documents_to_index = max_documents_to_index

    def retrieve(self, document: Union[str, List[str]], top_k: int = None, **kwargs) -> Union[List[str], List[List[str]]]:
        top_k = top_k or self._top_k
        return self._sampler.retrieve(document, top_k=top_k, **kwargs)
        
    def write_documents(self, documents: List[str], with_ids: bool = False, **kwargs) -> Optional[Dict[str, int]]:
        if self._max_documents_to_index:
            documents = documents[:self._max_documents_to_index]

        documents = list(set(documents))
            
        document_index = None
        if with_ids:
            document_index = self._sampler._build_document_mapping(documents)
            documents = [self._sampler._document_to_dict(d, doc_id=document_index[d]) for d in documents]
        self._sampler.write_documents(documents, **kwargs)
        return document_index
    
    def _build_document_mapping(self, documents: List[str]) -> Dict[str, int]:
        return self._sampler._build_document_mapping(documents)