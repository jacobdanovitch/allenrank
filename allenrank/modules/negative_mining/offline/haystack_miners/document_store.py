# https://github.com/deepset-ai/haystack/blob/master/haystack/document_store/elasticsearch.py

import logging
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)
hs_logger = logging.getLogger('haystack.retriever')
hs_logger.setLevel(logging.WARNING)

from allennlp.common import Registrable

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore


class HaystackDocumentStore(Registrable):
    pass

HaystackDocumentStore.register("faiss")(FAISSDocumentStore)
HaystackDocumentStore.register("sql")(SQLDocumentStore)
HaystackDocumentStore.register("memory")(InMemoryDocumentStore)
HaystackDocumentStore.register("es")(ElasticsearchDocumentStore)
