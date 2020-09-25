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
        document_store: HaystackDocumentStore,
        retriever: Lazy[HaystackRetriever],
        rebuild_index: bool = False, 
        bash_command: Union[str, List[str]] = None
    ):
        if rebuild_index:
            document_store.delete_all_documents(document_store.index)

        self.document_store = document_store
        self.retriever = retriever
        # self._query_fn = None

        if bash_command is not None:
            _exec_command(bash_command)

    def retrieve(self, document: Union[str, Dict[str, str]], top_k: int = 10, **kwargs) -> List[str]:
        if not self.initialized:
            raise RuntimeError("Miner is not yet initialized. Try calling `miner.write_documents` to add documents before retrieval.")
        
        results = [r.text for r in self.retriever.retrieve(document, top_k=top_k+1, **kwargs) if r.text != document]
        return results[:top_k]
        
    def write_documents(self, documents: Union[List[str], Dict[str, any], pd.Series], **kwargs) -> None:
        if isinstance(documents[0], dict):
            assert all(isinstance(d, dict) for d in documents), "Some documents were passed as dictionaries but not others."
        elif isinstance(documents, (pd.Series, pd.DataFrame)):
            documents = documents.drop_duplicates()
            apply_kwargs = dict(axis=1) if isinstance(documents, pd.DataFrame) else {}
            documents = documents.apply(self._document_to_dict, **apply_kwargs).values.to_list()
        else:
            documents = self.documents_to_dicts(documents)

        # documents = [Document(text=d['text'], id=str(d['id'])) for d in documents]
        
        self.document_store.write_documents(documents, **kwargs)
        self.retriever = self.retriever.construct(document_store=self.document_store)

    @property
    def initialized(self):
        return not isinstance(self.retriever, Lazy)



def _exec_command(bash_command: Union[str, List[str]]):
    from subprocess import Popen, PIPE

    if isinstance(bash_command, list):
        processes = [_exec_command(c) for c in bash_command]
        for p in processes:
            p.wait()
        
        return processes
    
    process = Popen(bash_command) # , stdout=PIPE, stderr=PIPE)
    return process