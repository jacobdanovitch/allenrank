import os, sys
import pandas as pd

from allennlp.common.file_utils import cached_path
from sklearn.model_selection import train_test_split

ROOT_URL = 'https://github.com/microsoft/MIMICS/raw/master/data/MIMICS-{}.tsv'

def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset', default='ClickExplore', choices=['Click', 'ClickExplore', 'Manual'])
    parser.add_argument('-o', '--output_path', type=os.path.abspath, default='/tmp/allenrank/data')

    return parser.parse_args()

def save(df, fp):
    print(f'{fp}: {df.shape}')
    df.to_csv(fp, sep='\t')

def main(
    url: str,
    output_path: str,
):
    os.makedirs(output_path, exist_ok=True)
    
    df = pd.read_csv(cached_path(url), sep='\t')
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    train, valid = train_test_split(train, test_size=0.2, random_state=42)
    
    print(f'Saving to {output_path}.')
    
    save(train, os.path.join(output_path, 'train.tsv'))
    save(valid, os.path.join(output_path, 'valid.tsv'))
    save(test, os.path.join(output_path, 'test.tsv'))

if __name__ == '__main__':
    args = parse_args()
    
    url = ROOT_URL.format(args.dataset)
    output_path = os.path.join(args.output_path, f'mimics-{args.dataset.lower()}')

    main(url=url, output_path=output_path)