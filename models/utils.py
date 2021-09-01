import pandas as pd
from os import path
import copy
from gensim.models import KeyedVectors
import gensim.downloader as api
import json
import numpy as np

def save_predictions(data,
                     preds,
                     split: str,
                     grain: str,
                     modality: str,
                     mode: str,
                     math_mode:str,
                     primary_model_name: str,
                     secondary_model_name: str,
                     filtered_on: int):
    assert len(data) == len(preds), 'You must pass data and predictions that are of the same length!'
    filename = \
        f'{modality}_{split}_{grain}_{mode}_{math_mode}_' \
        f'{primary_model_name}_{secondary_model_name}_filtered_on_{filtered_on}'

    true_labels = list(data['nc_type'])
    compounds = list(data['compound'])
    predictions_dict = {'compounds': compounds, 'true_labels': true_labels, 'predicted_labels' : list(preds)}
    model_path = f'../results/{filename}.json'
    with open(model_path, 'w') as outfile:
        json.dump(predictions_dict, outfile)


def combine_vectors(vec1, vec2, mode: str):
    if mode == 'add':
        return vec1 + vec2
    elif mode == 'multiply':
        return np.multiply(vec1, vec2)
    elif mode == 'concatenate':
        return np.concatenate((vec1, vec2))
    else:
        raise ValueError('mode of vector combination is not valid')

def reduce_data(data, based_on):
    #todo her skal det komme en funksjon som kan ta et datasett
    # og redusere det til ish-størrelsen på et annet datasett, med samme (ca.) balanse av klasser
    # TODO HVORFOR?? fordi jeg vil sjekke om lexical split-greia enkli bare er verre pga mindre train data:)
    #TODO SKAL DENNE EGENTLIG VÆRE HER I DET HELE TATT??
    assert(len(data) > len(based_on)), \
        'Length of your main data must be longer than ' \
        'the length you want to reduce it to (length of the based_on data).'

def load_data(data_path: str, part_of_dataset: str):
    assert(part_of_dataset in ['train', 'test', 'val']), \
        'Parameter part_of_dataset must be either train, test, or val.'
    data = pd.read_csv(path.join(data_path, f'{part_of_dataset}.tsv'), sep='\t', header=None)
    data.columns = ['nc_mod', 'nc_head', 'nc_type']
    data['compound'] = data['nc_mod'] + " " + data['nc_head']
    return data

# Removes words from the dataframe whose mod, head, or combined compound doesn't exist in the gensim model
def remove_non_existing(data: pd.DataFrame, model, filter_on: list):
    possible_to_filter_on = ['mod', 'head', 'compound']
    assert len(filter_on) > 0, 'Parameter filter_on must be a non-empty list'
    for word in filter_on:
        assert word in possible_to_filter_on, \
            f'Parameter filter_on must be a list containing at least one of the following: {possible_to_filter_on}'
    def compound_exists_in_model(compound):
        one_word_compound = ''.join(compound.split())
        underscore_compound = '_'.join(compound.split())
        hyphenated_compound = '-'.join(compound.split())

        if compound in model \
                or underscore_compound in model \
                or one_word_compound in model \
                or hyphenated_compound in model:
            return True
        else: return False

    def word_exists_in_model(word):
        if word in model:
            return True
        else:
            return False

    new_data = copy.deepcopy(data)
    if 'compound' in filter_on:
        new_data = new_data[new_data['compound'].map(compound_exists_in_model)]
    if 'mod' in filter_on:
        new_data = new_data[new_data['nc_mod'].map(word_exists_in_model)]
    if 'head' in filter_on:
        new_data = new_data[new_data['nc_head'].map(word_exists_in_model)]
    return new_data


def get_model(model_name: str):
    if model_name == 'word2vec':
        model = KeyedVectors.load_word2vec_format('../data/resources/GoogleNews-vectors-negative300.bin', binary=True)
    elif model_name == 'glove':
        model = api.load('glove-wiki-gigaword-300')
    #TODO fiks dette model_name-greiene
    elif model_name == 'ResNet_10':
        with open('../data/resources/word_to_img_vec_ResNet152V2_10_pca_300.json', 'r') as file:
            model = json.load(file)
    elif model_name == 'ResNet_100':
        with open('../data/resources/word_to_img_vec_ResNet152V2_100_pca_300.json', 'r') as file:
            model = json.load(file)
    elif model_name == 'ResNet_10_norm':
        with open('../data/resources/word_to_img_vec_ResNet152V2_10_pca_300_norm.json', 'r') as file:
            model = json.load(file)
    elif model_name == 'ResNet_100_norm':
        with open('../data/resources/word_to_img_vec_ResNet152V2_100_pca_300_norm.json', 'r') as file:
            model = json.load(file)
    elif model_name == 'ResNet_10_150_norm':
        with open('../data/resources/word_to_img_vec_ResNet152V2_10_pca_150_norm.json', 'r') as file:
            model = json.load(file)
    elif model_name =='ResNet_norm_300':
        with open('../data/resources/word_to_img_vec_ResNet152V2_10_norm_pca_300.json', 'r') as file:
            model = json.load(file)
    elif model_name == 'imagined':
        with open('../data/resources/imagined_vectors_resnet10_300_tanh_full.json', 'r') as file:
            model = json.load(file)
    else:
        raise ValueError('Invalid model name!')
    return model