# https://github.com/deepset-ai/haystack/blob/master/haystack/document_store/elasticsearch.py

from typing import Union, List, Dict, Optional
from overrides import overrides

import json
from string import Template
from copy import deepcopy

import logging
es_logger = logging.getLogger('elasticsearch')
es_logger.setLevel(logging.WARNING)
hs_logger = logging.getLogger('haystack.retriever')
hs_logger.setLevel(logging.WARNING)

from allennlp.common import Params, Registrable
from allenrank.modules.negative_mining.offline.offline_miner import OfflineNegativeMiner

from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.sql import SQLDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore


_CUSTOM_ELASTICSEARCH_QUERY_BASE = Template("""{
    "query": { 
        "bool": { 
            "should": [{"multi_match": { 
                    "query": "${question}",
                    "type": "most_fields", 
                    "fields": ${search_fields}
                }
            }], 
            "must_not": [
                { "ids": {
                        "values": ${document_ids}
                    }
                }
            ]
        } 
    }
}""")

class HaystackDocumentStore(Registrable):
    pass

HaystackDocumentStore.register("faiss")(FAISSDocumentStore)
HaystackDocumentStore.register("sql")(SQLDocumentStore)
HaystackDocumentStore.register("memory")(InMemoryDocumentStore)

@HaystackDocumentStore.register("haystack_elasticsearch", constructor='from_params')
class HaystackElasticSearchDocumentStore(ElasticsearchDocumentStore):
    _automatic_deduplication = True

    def __init__(self, rebuild_index: bool = False, validation: bool = False, index: str = 'document', **kwargs):
        if validation:
            index = f'validation_{index}'
        super().__init__(index=index, **kwargs)

        if rebuild_index:
            self.delete_all_documents(self.index)

        self._custom_query_template = Template(_CUSTOM_ELASTICSEARCH_QUERY_BASE.safe_substitute(
            search_fields=json.dumps(self.search_fields))
        )
    
    @classmethod
    def from_params(cls, params: Params, **extras):
        kwargs = deepcopy(params).as_dict()
        return cls(**kwargs)

    @overrides
    def query(self, query: Optional[Union[str, Dict[str, str]]], **kwargs):
        if isinstance(query, dict):
            document_ids = query.get('id', [])
            if not isinstance(document_ids, list):
                document_ids = [document_ids]

            custom_query = self._custom_query_template.safe_substitute(document_ids=json.dumps(document_ids))
            kwargs.setdefault('custom_query', custom_query)
        kwargs.setdefault('index', self.index)
        return super().query(query, **kwargs)



