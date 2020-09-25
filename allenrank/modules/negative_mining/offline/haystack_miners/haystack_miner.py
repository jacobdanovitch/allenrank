from typing import Union, List, Dict, Any
import pandas as pd

from haystack.schema import Document

from allennlp.common import Lazy

from allenrank.modules.negative_mining.offline.offline_miner import OfflineNegativeMiner
from allenrank.modules.negative_mining.offline.haystack_miners.document_store import HaystackDocumentStore
from allenrank.modules.negative_mining.offline.haystack_miners.retriever import HaystackRetriever

@OfflineNegativeMiner.register("haystack_miner")
class HaystackMiner(OfflineNegativeMiner):
    def __init__(
        self, 
        document_store: HaystackDocumentStore = None,
        retriever: Lazy[HaystackRetriever] = None,
        bash_command: Union[str, List[str]] = None
    ):
        self.document_store = document_store
        self.document_store.delete_all_documents()
        self.retriever = retriever
        self._query_fn = None

        if bash_command is not None:
            _exec_command(bash_command)
    
    def _set_query_function(self):
        query_fn = getattr(self.document_store, 'query', None)
        if query_fn is None:
            retriever_fn = self.retriever and self.retriever.retrieve
            if retriever_fn:
                self._query_fn = retriever_fn
                return
            
            self._query_fn = getattr(self.document_store, 'query_by_embedding', None)
            assert self._query_fn is not None, f"Unable to find query function on {type(self.document_store).__name__}"
            return
        
        self._query_fn = query_fn

    def retrieve(self, document: Union[str, Dict[str, str]], top_k: int = 10, **kwargs) -> List[str]:
        if not self.initialized:
            raise RuntimeError("Miner is not yet initialized. Try calling `miner.write_documents` to add documents before retrieval.")

        if not getattr(self.document_store, '_automatic_deduplication', False):
            top_k = top_k+1

        
        results = [r.text for r in self._query_fn(document, top_k=top_k, **kwargs) if r.text != document]
        assert len(results) <= top_k

        return results
        
    def write_documents(self, documents: Union[List[str], Dict[str, any], pd.Series], **kwargs) -> None:
        if isinstance(documents, (pd.Series, pd.DataFrame)):
            documents = documents.drop_duplicates()
            apply_kwargs = dict(axis=1) if isinstance(documents, pd.DataFrame) else {}
            documents = documents.apply(self._document_to_dict, **apply_kwargs).values.to_list()
        elif isinstance(documents[0], dict):
            pass
        else:
            documents = self.documents_to_dicts(documents)

        # documents = set(documents)
        documents = [Document(text=d['text'], id=str(d['id'])) for d in documents]
        # all_ids = [d.id for d in documents]
        # assert len(all_ids) == len(set(all_ids))
        
        self.document_store.write_documents(documents, **kwargs)
        self.retriever = self.retriever.construct(document_store=self.document_store)
        # raise ValueError(elf.retriever)
        self._set_query_function()

    @property
    def initialized(self):
        return self._query_fn is not None



def _exec_command(bash_command: Union[str, List[str]]):
    from subprocess import Popen, PIPE

    if isinstance(bash_command, list):
        processes = [_exec_command(c) for c in bash_command]
        for p in processes:
            p.wait()
        
        return processes
    
    process = Popen(bash_command) # , stdout=PIPE, stderr=PIPE)
    return process