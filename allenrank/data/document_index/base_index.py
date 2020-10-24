from typing import List, Dict, Union

from allennlp.common.registrable import Registrable
from allennlp.common.util import JsonDict

class DocumentIndex(Registrable):
    def write_document(self, document: Union[JsonDict, str]) -> None:
        self.write_documents([document])

    def write_documents(self, documents: List[Union[JsonDict, str]]) -> None:
        raise NotImplementedError