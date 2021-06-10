from nltk.corpus import wordnet as wn
import pandas as pd
import argparse
from os import path

def get_synsets(data: pd.DataFrame, constituent='whole'):
    """
    :param data: pandas DataFrame containing columns nc_mod, nc_head, nc_type
    :param constituent: str, 'mod' for modifiers, 'head' for heads, 'whole' for entire compounds
    :return: #todo
    """
    for i, row in data.iterrows():
        pass

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t')
    data.columns = ['nc_mod', 'nc_head', 'nc_type']
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile_dir', type=str, required=True,
                        help='Directory where the train, test, and val files are')

    args = parser.parse_args()
    infile_dir = args.infile_dir

    train_data = load_data(path.join(infile_dir, 'train.tsv'))
    test_data = load_data(path.join(infile_dir, 'test.tsv'))
    val_data = load_data(path.join(infile_dir, 'val.tsv'))

if __name__ == '__main__':
    main()