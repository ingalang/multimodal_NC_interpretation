from models.full_add_composition import remove_non_existing, load_data, get_model
from itertools import product
import os
from gensim.models import KeyedVectors
import gensim.downloader as api
import json
import numpy as np
import pandas as pd

def get_model(model_name):
    if model_name == 'glove':
        model = api.load('glove-wiki-gigaword-300')
    elif model_name == 'word2vec':
        model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    elif model_name == 'ResNet-avg10':
        with open('word_to_img_vec_ResNet152V2_10_pca_300.json', 'r') as file:
            model = json.load(file)
    else:
        raise ValueError('Invalid model name')
    return model

def get_fulladd_train_support(data, model, name):
    print(f'Getting fullAdd training support for {name}')
    print(f'Original length of data: {len(data)}')
    filter_on = ['mod', 'head', 'compound']
    new_data = remove_non_existing(data=data, model=model, filter_on=filter_on)
    print(f'New length of data, after removing all \n'
          f'samples that do not have {filter_on} in the model: {len(new_data)}')
    print()
    print()


def get_class_distribution(data, include_examples=True):
    print(len(data))
    labels = pd.unique(data['nc_type'])
    print(labels)

    for l in labels:
        samples = data[data['nc_type'] == l]
        num_samples = len(samples)
        print(f'Label {l} has {num_samples} samples.')

        if include_examples:
            print('Some examples are:')
            print(samples.head())
        print()


def main():
    grains = ['coarse', 'fine']
    splits = ['random', 'lexical_full', 'lexical_mod', 'lexical_head']

    # model_name = 'ResNet-avg10'
    # model = get_model(model_name)


    data_dir = '../../data/Tratz_2011_data_comp_binary'

    for combo in product(grains, splits):
        suffix = f'tratz2011_cb_{combo[0]}_grained_{combo[1]}'
        print(f'{combo[0]}_grained_{combo[1]}')
        data_subdir = os.path.join(data_dir, suffix)
        train = load_data(data_subdir, 'train')
        test = load_data(data_subdir, 'test')
        val = load_data(data_subdir, 'val')

        full_data = (train.append(test, ignore_index=True)).append(val, ignore_index=True)

        get_class_distribution(full_data)

        # get_fulladd_train_support(train, model, f'{suffix} train')
        # get_fulladd_train_support(test, model, f'{suffix} test')
        # get_fulladd_train_support(val, model, f'{suffix} val')

if __name__ == '__main__':
    main()