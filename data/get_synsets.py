from nltk.corpus import wordnet as wn
import pandas as pd
import argparse
from os import path
from doltpy.cli import Dolt
from doltpy.cli.read import read_pandas_sql
import os
import time


def get_synsets(data: pd.DataFrame, constituent='both'):
    """
    :param data: pandas DataFrame containing columns nc_mod, nc_head, nc_type
    :param constituent: str, 'mod' for modifiers, 'head' for heads, 'both' for both, and 'whole' for entire compounds
    :return: #todo
    """
    ## HAR TESTET METODENE: filtering med query først er definitivt raskest!!!!

    dolt_dir ='imagenet/image-net'
    dolt = Dolt(dolt_dir)

    ### METODE 1: filtrere på ord etter at man har fått df-en
    # filter_df_start = time.time()
    # query = 'select * from images_synsets where synset_type=\'n\''
    # df_one = read_pandas_sql(dolt, query)
    #
    # wordlist = ['dog', 'cat', 'carrot', 'squirrel', 'house', 'dolphin', 'cloud', 'umbrella', 'fox', 'rabbit']
    # url_list = []
    #
    # for word in wordlist:
    #     synsets = wn.synsets(word, pos='n')
    #     print(synsets)
    #     print(synsets[0].offset())
    #     synset_id = f'n{synsets[0].offset()}'
    #     synset_subset = df_one[df_one['synset_id'] == synset_id]
    #     url_list.append([row['image_url'] for i, row in synset_subset.iterrows()])
    # print(url_list)
    # print(len(url_list))
    # filter_df_end = time.time()
    # print(f'Filtering on the df took {filter_df_end-filter_df_start} seconds')

    ### METODE 2: filtrere på ord i queryen og få en df per ord
    filter_query_start = time.time()
    wordlist_two = ['cat', 'carrot', 'squirrel']
    url_list_two = []

    for word in wordlist_two:
        word_query = f'select * from words_synsets where word=\'{word}\' and synset_type=\'n\''
        df_synsets = read_pandas_sql(dolt, word_query)
        print(df_synsets.head())
        synset_ids = [str(row['synset_id']) for i, row in df_synsets.iterrows()]
        synset_ids = list(map(lambda x: f'0{x}' if len(x)<8 else x, synset_ids))
        print(synset_ids)
        print(f'SYNSET IDs FOR {word}: {synset_ids}')
        first_id = str(synset_ids[0])
        rest_of_synsets = [str(s) for s in synset_ids[1:]]
        img_synset_query = f'select * from images_synsets where synset_type=\'n\' and (synset_id=\'{first_id}\''
        for syns in rest_of_synsets:
            img_synset_query +=  f' or synset_id=\'{syns}\''
        img_synset_query += ')'
        print(img_synset_query)
        df_image_urls = read_pandas_sql(dolt, img_synset_query)
        df_image_urls['synset_id'] = list(map(lambda x: f'0{x}' if len(x)<8 else x, df_image_urls['synset_id']))
        print(f'{word} RESULTS:')
        print(len(df_image_urls))
        sucessful_synsets = pd.unique(df_image_urls['synset_id'])
        print(f'SUCCESSFUL SYNSETS FOR {word}: {sucessful_synsets}')
        for i, row in df_image_urls.iterrows():
            url = row['image_url']
            print(url)
    print(type(url_list_two))
    print(url_list_two)
    print(len(url_list_two))
    filter_query_end = time.time()

    print(f'Filtering by query took {filter_query_end - filter_query_start} seconds')



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

    get_synsets(train_data)


if __name__ == '__main__':
    main()