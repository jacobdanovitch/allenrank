# https://github.com/deepset-ai/haystack/blob/master/haystack/document_store/elasticsearch.py

from allenrank.data.document_index.base_index import DocumentIndex

import logging
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)
hs_logger = logging.getLogger('haystack.retriever')
hs_logger.setLevel(logging.WARNING)

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore


class HaystackDocumentStore(DocumentIndex):
    pass

HaystackDocumentStore.register("faiss")(FAISSDocumentStore)
HaystackDocumentStore.register("sql")(SQLDocumentStore)
HaystackDocumentStore.register("memory")(InMemoryDocumentStore)
HaystackDocumentStore.register("es")(ElasticsearchDocumentStore)
