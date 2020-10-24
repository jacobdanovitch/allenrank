# Modelled after https://github.com/deepset-ai/haystack/blob/master/haystack/schema.py

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Union, Tuple
from allennlp.common.util import JsonDict

import torch
import numpy as np

Array = Union[torch.Tensor, np.array, List[float]]

@dataclass
class Text:
    text: str
    id: Union[str, int] = None,
    features: Dict[str, Array] = field(default_factory=dict, repr=False)

    def __hash__(self):
        return self.id

    @classmethod
    def from_dict(cls, d: JsonDict, field_map: Dict[str, str] = {}):
        return cls(**translate_dict(d, field_map))

    def to_dict(self, field_map: Dict[str, str] = {}):
        return translate_dict(asdict(self), field_map)

def translate_dict(d: JsonDict, field_map: Dict[str, str] = {}):
    return {field_map.get(k, k): v for (k, v) in d.items()}