# https://github.com/deepset-ai/haystack/blob/master/haystack/document_store/elasticsearch.py

from typing import Union, List, Dict, Optional
from overrides import overrides


from allennlp.common import Params, Registrable

from haystack.retriever.sparse import TfidfRetriever, ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever

from allenrank.modules.negative_mining.offline.haystack_miners.document_store import HaystackDocumentStore

class HaystackRetriever(Registrable):
    # @classmethod
    # def from_partial_objects(
    #     cls,
    #     document_store: HaystackDocumentStore,
    #     **kwargs
    # ):
    #     document_store = document_store.construct()
    #     return cls(document_store=document_store, **kwargs)
    @staticmethod
    def default(document_store: HaystackDocumentStore) -> "HaystackRetriever":
        return HaystackRetriever.from_params(document_store=document_store, params=Params({}))
        

@HaystackRetriever.register('tfidf')
class HaystackTfIdfRetriever(HaystackRetriever, TfidfRetriever):
    def __init__(self, document_store: HaystackDocumentStore):
        super().__init__(document_store=document_store)


HaystackRetriever.register('es')(ElasticsearchRetriever)
HaystackRetriever.register('embedding')(EmbeddingRetriever)
HaystackRetriever.register('dpr')(DensePassageRetriever)