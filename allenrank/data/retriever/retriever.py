from typing import List, Dict, Union
from overrides import overrides

from allennlp.common.registrable import Registrable

from allennlp.common.util import JsonDict
from allennlp.data import Instance

from allenrank.data.document_index.base_index import DocumentIndex

class Retriever(Registrable):
    def __init__(self, index: DocumentIndex):
        self._index = index

    def get_document_by_id(self, doc_id: Union[str, int]) -> Union[JsonDict, str]:
        raise NotImplementedError

    def get_documents_by_ids(self, doc_ids: List[Union[str, int]]) -> List[Union[JsonDict, str]]:
        raise NotImplementedError
    
    def retrieve(self, query: str, top_k: int = 10) -> List[JsonDict]:
        raise NotImplementedError

    def batch_retrieve(self, queries: List[str], top_k: int = 10) -> List[List[JsonDict]]:
        # retrieve = functools.partial(self.retrieve, top_k=top_k)
        # yield from map(retrieve, queries)
        for query in queries:
            for res in self.retrieve(query, top_k=top_k):
                yield res