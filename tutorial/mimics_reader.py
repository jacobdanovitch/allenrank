from typing import Dict, List
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allenrank.dataset_readers import ListwiseRankingReader

import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

@DatasetReader.register("mimics")
class MIMICSDatasetReader(ListwiseRankingReader):
    @overrides
    def _read(self, file_path: str):
        df = read_df(file_path).rename(columns={'options': 'documents'})
        for row in df.to_dict(orient='records'):
            row['query'] = (row.pop('query'), row.pop('question'))
            yield self.text_to_instance(**row)


def read_df(file_path: str, **kwargs):
    logger.info("Reading instances from lines in file at: %s", file_path)

    df = pd.read_csv(cached_path(file_path), sep='\t', **kwargs)
    
    _options_columns = [f'option_{i}' for i in range(1, 6)] # option_1, ..., option_5
    _label_columns = df.filter(regex=r"option\_.*\_\d", axis=1).columns.tolist() # option_label_1, ..., option_label_5
    
    columns = ['query','question', *_options_columns, *_label_columns]
    df = df[columns]

    df['options'] = df[_options_columns].fillna('').values.tolist()
    df['labels'] = df[_label_columns].values.tolist()

    df = df.drop(columns=[*_options_columns, *_label_columns])
    return df