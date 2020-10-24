from typing import List, Dict, Tuple

from allenrank.data.document_index.base_index import DocumentIndex

from pyserini.search import SimpleSearcher

# @DocumentIndex.register('anserini', exist_ok=True)
# class AnseriniDocumentIndex(DocumentIndex):
#     def __init__(self, collection: str):
#         self._searcher = SimpleSearcher.from_prebuilt_index(collection)


DocumentIndex.register('anserini', exist_ok=True, constructor='from_prebuilt_index')(SimpleSearcher)