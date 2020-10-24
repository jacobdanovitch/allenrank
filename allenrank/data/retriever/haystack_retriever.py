# https://github.com/deepset-ai/haystack/blob/master/haystack/document_store/elasticsearch.py

from typing import Union, List, Dict, Optional
from overrides import overrides


from allennlp.common import Params, Registrable

from haystack.retriever.sparse import TfidfRetriever, ElasticsearchRetriever
from haystack.retriever.dense import EmbeddingRetriever, DensePassageRetriever

from allenrank.data.document_index.haystack_index import HaystackDocumentStore
from allenrank.data.retriever.retriever import Retriever

class HaystackRetriever(Retriever):
    # pass
    default_implementation = 'es'


# Can't just do HaystackRetriever.register('tfidf')(TfidfRetriever);
# Lazy[T] can't accept positionals which the retriever classes require,
# but somehow inheriting from both fixes that      
@HaystackRetriever.register('tfidf')
class HaystackTfIdfRetriever(HaystackRetriever, TfidfRetriever):
    pass

@HaystackRetriever.register('es')
class HaystackElasticsearchRetriever(HaystackRetriever, ElasticsearchRetriever):
    pass


@HaystackRetriever.register('embedding')
class HaystackEmbeddingRetriever(HaystackRetriever, EmbeddingRetriever):
    pass

@HaystackRetriever.register('dpr')
class HaystackDensePassageRetriever(HaystackRetriever, DensePassageRetriever):
    pass