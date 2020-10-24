from typing import List, Dict, Union, Tuple, Iterable
from collections import defaultdict
import string

from allennlp.common.util import JsonDict
from allennlp.data import Vocabulary
from allennlp.data.tokenizers import Tokenizer, WhitespaceTokenizer

from allenrank.data.document_index.base_index import DocumentIndex
from allenrank.data.schema import Text, Array

InputText = Union[str, JsonDict]

# based on http://www.dalkescientific.com/writings/diary/archive/2012/06/10/inverted_index_library.html
class InvertedIndex(DocumentIndex):
    def __init__(self):
        self.inverted_indices = defaultdict(set)
        self._document_map = defaultdict(dict)
        self._count = 0

    def text_to_features(self, text: JsonDict) -> Array:
        raise NotImplementedError

    def batch_text_to_features(self, texts: List[InputText], lazy: bool = False) -> Iterable[Array]:
        features = (self.text_to_features(t) for t in texts)
        if lazy:
            features = list(features)
        return features

    def _inputs_to_dicts(self, texts: List[InputText]) -> List[JsonDict]:
        for document in texts:
            if isinstance(document, str):
                document = {'text': document, 'id': len(self._document_map)}
            yield document

    def write_documents(self, documents: List[InputText]) -> None:
        for document in self._inputs_to_dicts(documents):
            features = self.text_to_features(document['text'])
            document = Text(**document, features={'indices': features})
            
            for feat in features:
                self.inverted_indices[feat].add(document.id)
                self._document_map[document.id] = document

    def search(self, query: str):
        features = self.text_to_features(query)
        terms = (self.inverted_indices.get(feature) for feature in features)
        terms = list(filter(None, terms))
        terms.sort(key=len)
        return set.intersection(*terms)

@DocumentIndex.register('term_index', exist_ok=True)
class TermBasedInvertedIndex(InvertedIndex):
    def __init__(
        self, 
        vocab: Vocabulary = Vocabulary(),
        tokenizer: Tokenizer = WhitespaceTokenizer()
    ):
        super().__init__()
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._translate = str.maketrans({c: ' ' for c in string.punctuation})

    def text_to_features(self, text: str) -> List[int]:
        tokens = [t.text for t in self._tokenizer.tokenize(text)]
        return self._vocab.add_tokens_to_namespace(tokens)

if __name__ == '__main__':
    import sys
    _, corpus_path, query = sys.argv[:3]

    with open(corpus_path) as f:
        corpus = f.read().split('\n')

    index = TermBasedInvertedIndex()
    index.write_documents(corpus)

    print(query)
    for d in index.search(query):
        print(d) # corpus[d]
