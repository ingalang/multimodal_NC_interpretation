import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import os
import json
from itertools import product
import numpy as np
from scipy import stats
from prettytable import PrettyTable
import copy
import matplotlib.pyplot as plt
from collections import Counter
from models.utils import get_model, load_data
import itertools

# This function was adapted from Rotem Dror's repository
# https://github.com/rtmdrr/testSignificanceNLP/blob/master/testSignificance.py
def calculateContingency(data_A, data_B):
    assert(len(data_A) == len(data_B)), 'List a and list b must be of the same length!'
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(len(data_A)):
        if (data_A[i] == 1 and data_B[i] == 1):
            ABrr = ABrr + 1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])

# This function was borrowed from Rotem Dror's repository
# https://github.com/rtmdrr/testSignificanceNLP/blob/master/testSignificance.py
def mcNemar(table):
    statistic = float(np.abs(table[0][1] - table[1][0])) ** 2 / (table[1][0] + table[0][1])
    pval = 1 - stats.chi2.cdf(statistic, 1)
    return pval

def convert_preds_to_binary_list(preds_dict):
    true_labels = preds_dict['true_labels']
    predicted_labels = preds_dict['predicted_labels']
    binary_list = [1 if true == pred else 0 for true, pred in zip(true_labels, predicted_labels)]
    return binary_list

def calculate_mcnemar_statistic(preds_a, preds_b):
    binary_list_a = convert_preds_to_binary_list(preds_a)
    binary_list_b = convert_preds_to_binary_list(preds_b)

    contingency_table = calculateContingency(binary_list_a, binary_list_b)
    mcnemar_p = mcNemar(contingency_table)
    return mcnemar_p


def get_file(args: dict):
    split, grain, modality, mode, math_mode, primary_model_name, secondary_model_name, filtered_on = \
        args['split'], args['grain'], args['modality'], args['mode'], args['math_mode'], args['primary_model_name'], \
        args['secondary_model_name'], args['filtered_on']
    if secondary_model_name == 'ResNet_10_norm':
        alt_secondary_model_name = 'ResNet_10'
    else:
        alt_secondary_model_name = 'ResNet_10_norm'
    filename = \
        f'{modality}_{split}_{grain}_{mode}_{math_mode}_' \
        f'{primary_model_name}_{secondary_model_name}_filtered_on_{filtered_on}.json'
    alt_filename = f'{modality}_{split}_{grain}_{mode}_{math_mode}_' \
                   f'{primary_model_name}_{alt_secondary_model_name}_filtered_on_{filtered_on}.json'
    if os.path.exists(f'../results/{filename}'):
        with open(f'../results/{filename}', 'r') as file:
            preds_file = json.load(file)
    elif os.path.exists(f'../results/{alt_filename}'):
        with open(f'../results/{alt_filename}', 'r') as file:
            preds_file = json.load(file)
    else:
        raise FileNotFoundError(f'could not find {filename} nor {alt_filename}')
    return preds_file

def get_file_metadata(file_path, format='bert'):
    #TODO du kan legge til at man kan 'infer format' aka at den finner ut formatet ved å lese hva det står i filnavnet
    basename = os.path.splitext(os.path.basename(file_path))[0]
    print(basename)
    file_features = [str(f) for f in basename.split('_')]
    if format == 'bert':
        metadata_categories = ['dataset', 'model', 'split', 'grain', 'epochs', 'batch_size', 'max_len', 'dual_seq', 'emb_comp']
        if len(file_features) > len(metadata_categories):
            i = file_features.index('lexical')
            file_features[i] = 'lexical_' + str(file_features[i+1])
            del file_features[i+1]
        return dict(zip(metadata_categories, file_features))
    else:
        #TODO når du har laget skriptet for dissect-greiene, kan du ha en modus for det filformatet også.
        metadata_categories = ['dataset', 'model', 'split', 'grain', 'epochs', 'batch_size', 'max_len', 'dual_seq',
                               'emb_comp']

def print_single_confusion_matrix(data: pd.DataFrame):
    true = data.nc_type
    pred = data.predicted_id
    all_labels = pd.concat([true, pred], axis=0)
    unique_labels = pd.unique(all_labels)
    print(unique_labels)
    print(confusion_matrix(y_true=true, y_pred=pred))

def print_classification_report(data: pd.DataFrame):
    true = data.nc_type
    pred = data.predicted_id
    all_labels = pd.concat([true, pred], axis=0)
    unique_labels = pd.unique(all_labels)
    print(unique_labels)
    print(classification_report(true, pred, labels=unique_labels, zero_division=0))

def print_report_headline(args_a: dict, args_b: dict):
    model_descriptions = []
    for args in [args_a, args_b]:
        split, grain, modality, mode, math_mode, primary_model_name, secondary_model_name, filtered_on = \
            args['split'], args['grain'], args['modality'], args['mode'], args['math_mode'], args['primary_model_name'], \
            args['secondary_model_name'], args['filtered_on']
        model_name = f'{modality}_{split}_{grain}_{mode}_{math_mode}_{primary_model_name}_' \
                     f'{secondary_model_name}_filtered_on_{filtered_on}'
        model_descriptions.append(model_name)
    print(f'MODEL A: {model_descriptions[0]}')
    print(f'MODEL B: {model_descriptions[1]}\n')

def print_general_comparison(preds_a, preds_b):
    accuracies = [accuracy_score(preds['true_labels'], preds['predicted_labels']) for preds in [preds_a, preds_b]]
    macro_f1s = [f1_score(preds['true_labels'], preds['predicted_labels'], average='macro') for preds in [preds_a, preds_b]]
    micro_f1s = [f1_score(preds['true_labels'], preds['predicted_labels'], average='micro') for preds in [preds_a, preds_b]]
    weighted_f1s = [f1_score(preds['true_labels'], preds['predicted_labels'], average='weighted') for preds in [preds_a, preds_b]]

    t = PrettyTable(['Metric', 'Model A', 'Model B'])
    t.add_row(['Accuracy'] + accuracies)
    t.add_row(['Macro F1'] + macro_f1s)
    t.add_row(['Micro F1'] + micro_f1s)
    t.add_row(['Weighted F1'] + weighted_f1s)
    print(t)

def plot_per_relation_f1_differences(preds_a, preds_b, model_a_name='w2v', model_b_name='w2v+ResNet10', title=''):
    labels = np.unique(list(preds_a['true_labels']) + list(preds_a['predicted_labels'])
                       + list(preds_b['true_labels']) + list(preds_b['predicted_labels']))
    a_scores = []
    b_scores = []
    model_a_report = classification_report(preds_a['true_labels'], preds_a['predicted_labels'], output_dict=True,
                                           zero_division=0)
    model_b_report = classification_report(preds_b['true_labels'], preds_b['predicted_labels'], output_dict=True,
                                           zero_division=0)
    label_counter = Counter(preds_a['true_labels'])
    sorted_labels = sorted(label_counter, key=label_counter.get, reverse=True)

    for l in sorted_labels:
        a_scores.append(model_a_report[l]['f1-score'])
        b_scores.append(model_b_report[l]['f1-score'])
    # sorted_labels = [label[:13] if len(label) > 20 else label for label in sorted_labels]
    # plt.figure(figsize=(20,4))
    # plt.plot(sorted_labels, a_scores, label=model_a_name, marker='s')
    # plt.plot(sorted_labels, b_scores, label=model_b_name, marker='o')
    # if len(sorted_labels) > 12:
    #     plt.xticks(rotation=85, fontsize=10)
    #     plt.subplots_adjust(bottom=0.45)
    # else:
    #     plt.xticks(rotation=45, fontsize=10)
    #     plt.subplots_adjust(bottom=0.25)
    # plt.title(title)
    # plt.legend()
    #
    # plt.savefig(f'{title}.png', dpi=300)
    # plt.show()

    sorted_labels = [label[:13] if len(label) > 20 else label for label in sorted_labels]
    x = np.arange(len(sorted_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16,4))
    ax.bar(x - width / 2, a_scores, width, label=model_a_name)
    ax.bar(x + width / 2, b_scores, width, label=model_b_name)

    ax.set_ylabel('F1 scores')
    ax.set_title(title)
    ax.set_xticks(x)
    if len(sorted_labels) > 12:
        ax.set_xticklabels(sorted_labels, rotation=85, fontsize=8)
    else:
        ax.set_xticklabels(sorted_labels, rotation=85, fontsize=10)
    ax.legend()

    #ax.bar_label(rects1, padding=3)
    #ax.bar_label(rects2, padding=3)

    plt.subplots_adjust(left=0.25)

    fig.tight_layout()

    plt.savefig(f'{title}.png', dpi=300)

    plt.show()

def plot_change_in_f1_by_image_availability(split, grain, preds_a, preds_b, image_model_name):
    image_model = get_model(image_model_name)

    def word_in_model(word):
        if word in image_model:
            return True
        else:
            return False

    data_path = f'../data/Tratz_2011_data_comp_binary/tratz2011_cb_{grain}_grained_{split}'
    test_data = load_data(data_path, 'test')

    labels = np.unique(list(preds_a['true_labels']) + list(preds_a['predicted_labels'])
                       + list(preds_b['true_labels']) + list(preds_b['predicted_labels']))

    model_a_report = classification_report(preds_a['true_labels'], preds_a['predicted_labels'], output_dict=True,
                                           zero_division=0)
    model_b_report = classification_report(preds_b['true_labels'], preds_b['predicted_labels'], output_dict=True,
                                           zero_division=0)

    head_ratios = []
    mod_ratios = []
    f1_differences = []

    for l in labels:
        label_compounds = test_data[test_data['nc_type'] == l]
        head_available = label_compounds[label_compounds['nc_head'].map(word_in_model)]
        mod_available = label_compounds[label_compounds['nc_mod'].map(word_in_model)]
        head_ratios.append(len(head_available)/len(label_compounds))
        mod_ratios.append(len(mod_available)/len(label_compounds))
        f1_differences.append(model_b_report[l]['f1-score']-model_a_report[l]['f1-score'])
    print(head_ratios)
    print(mod_ratios)
    print(f1_differences)
    print(labels)

    f1_differences, mod_ratios = zip(*sorted(zip(f1_differences, mod_ratios)))
    _, head_ratios = zip(*sorted(zip(f1_differences, head_ratios)))
    print('head + mod ratios and f1 differences after sorting:')
    print(head_ratios)
    print(mod_ratios)
    print(f1_differences)

    plt.plot(f1_differences, head_ratios)
    plt.plot(f1_differences, mod_ratios)
    plt.show()

def print_per_relation_differences(preds_a, preds_b, metric='f1-score'):
    model_a_report = classification_report(preds_a['true_labels'], preds_a['predicted_labels'], output_dict=True, zero_division=0)
    model_b_report = classification_report(preds_b['true_labels'], preds_b['predicted_labels'], output_dict=True, zero_division=0)
    print(model_a_report)
    print(model_b_report)
    t = PrettyTable(['Label', f'Model A {metric}', f'Model B {metric}', f'Difference in {metric}', 'support'])

    for label, report_dictionary in model_a_report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg', 'f1-score', 'support']:
            try:
                a_metric = report_dictionary[metric]
            except:
                a_metric = 0
            try:
                b_metric = model_b_report[label][metric]
            except:
                b_metric = 0

            f1_diff = b_metric - a_metric
            support = report_dictionary['support']

            t.add_row([label, float("{:.3f}".format(a_metric)), float("{:.3f}".format(b_metric)), float("{:.3f}".format(f1_diff)), support])

    print(t)

def print_classification_differences(preds_a, preds_b):
    labels = np.unique(list(preds_a['true_labels']) + list(preds_a['predicted_labels'])
                       + list(preds_b['true_labels']) + list(preds_b['predicted_labels']))
    cls_differences_dict = {l : {'ArBr' : [],
                                 'ArBw' : [],
                                 'AwBr' : [],
                                 'AwBw' : []} for l in labels}

    for i in range(len(preds_a['compounds'])):
        compound = preds_a['compounds'][i]
        true_label = preds_a['true_labels'][i]
        a_predicted = preds_a['predicted_labels'][i]
        b_predicted = preds_b['predicted_labels'][i]
        if a_predicted == true_label:
            if b_predicted == true_label:
                cls_differences_dict[true_label]['ArBr'].append(compound)
            else:
                cls_differences_dict[true_label]['ArBw'].append(compound)
        else:
            if b_predicted == true_label:
                cls_differences_dict[true_label]['AwBr'].append(compound)
            else:
                cls_differences_dict[true_label]['AwBw'].append(compound)

    for label, subdict in cls_differences_dict.items():
        print(f'{label}:')
        for k, v in subdict.items():
            print(k,': ', v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=False, default='all',
                        choices=['random', 'lexical_full', 'lexical_head', 'lexical_mod', 'all'],
                        help='What data split to use (random, lexical_full, lexical_head, lexical_mod, all)')
    parser.add_argument('--grain', type=str, required=False, default='all',
                        choices=['coarse', 'fine', 'all'],
                        help='What grain (fineness of relation classes) to use '
                             '(fine, with 37 classes; coarse, with 12 classes; or all, which gets both)')
    parser.add_argument('--modality', type=str, required=False, default='unimodal_text',
                        choices=['unimodal_text', 'unimodal_images', 'multimodal', 'baseline'],
                        help='Which modality was used')
    parser.add_argument('--mode', type=str, required=False, default='similar_first',
                        choices=['similar_first', 'opposite_first'],
                        help='Which mode was used for training and combining the multimodal vectors. \n'
                             'similar_first: composes textual and image vectors separately and then combines them. \n'
                             'opposite_first: combines textual mod vectors with image mod vectors, '
                             'and similarly for head vectors. Then combines the resulting multimodal mod & head vectors ')
    parser.add_argument('--math_mode', type=str, required=False, default='concatenate',
                        choices=['fulladd', 'multiply', 'concatenate'],
                        help='Math mode was used when combining vectors together. \n'
                             'Options: fulladd, multiply, concatenate \n'
                             'Default: fulladd')
    parser.add_argument('--text_model', type=str, required=False, default='word2vec',
                        choices=['word2vec', 'glove', 'imagined'],
                        help='What pre-trained model was used for textual embeddings. \n'
                             'Default: word2vec')
    parser.add_argument('--image_model', type=str, required=False, default='ResNet_10_norm',
                        choices=['ResNet_10', 'ResNet_100', 'ResNet_10_norm',
                                 'ResNet_100_norm', 'ResNet_10_150_norm',
                                 'ResNet_norm_300', 'imagined'],
                        help='What pre-computed embeddings to use for the images. \n'
                             'Options: ResNet_10, ResNet_100, ResNet_10_norm, ResNet_100_norm, ResNet_10_150_norm, imagined \n'
                             'Default: ResNet_10 (ResNet152V2 vectors averaged over 10 images per word, '
                             'and PCA-reduced to 300 dimensions.')
    parser.add_argument('--filter_data_by_model', type=int, required=False, default=0,
                        choices=[0, 1, 2, 3],
                        help='How to filter the data. \n'
                             'Type=int \n'
                             'Options: 0 means no filtering. 1 filters on the primary model '
                             '(which is the text model unless modality is unimodal_images). '
                             '2 filters on both the primary model and secondary model, '
                             'and 3 additionally filters on ResNet100 vectors '
                             '(since there are fewer of them than of the ResNet10 vectors).')
    parser.add_argument('--difference', type=str, required=False, default='modality:multimodal')
    args = parser.parse_args()
    split, grain, modality, mode, math_mode, text_model_name, image_model_name, filter_data_by_model, difference = \
        args.split, args.grain, args.modality, args.mode, args.math_mode, \
        args.text_model, args.image_model, args.filter_data_by_model, args.difference

    ### getting specifications for model a
    if modality=='unimodal_images':
        primary_model_name = image_model_name
        secondary_model_name = text_model_name
    else:
        primary_model_name = text_model_name
        secondary_model_name = image_model_name

    if split == 'all':
        splits = ['random', 'lexical_full', 'lexical_mod', 'lexical_head']
    else:
        splits = [split]
    if grain == 'all':
        grains = ['coarse', 'fine']
    else:
        grains = [grain]

    print('############ STATISTICS REPORT ############\n')

    for combo in product(splits, grains):
        split = combo[0]
        grain = combo[1]
        model_a_args = {'split': split,
                        'grain' : grain,
                        'modality' : modality,
                        'mode' : mode,
                        'math_mode' : math_mode,
                        'primary_model_name' : primary_model_name,
                        'secondary_model_name' : secondary_model_name,
                        'filtered_on' : filter_data_by_model}
        predictions_a = get_file(model_a_args)

        param_to_change, change_to = difference.split(':')

        model_b_args = copy.copy(model_a_args)
        model_b_args[param_to_change] = change_to

        if model_b_args['modality'] == 'unimodal_images':
            model_b_args['primary_model_name'] = image_model_name
            model_b_args['secondary_model_name'] = text_model_name

        predictions_b = get_file(model_b_args)

        ### printing report
        print_report_headline(model_a_args, model_b_args)
        print_general_comparison(predictions_a, predictions_b)
        mcnemar_p_value = calculate_mcnemar_statistic(predictions_a, predictions_b)
        alpha = 0.05

        model_a_f1 = f1_score(predictions_a['true_labels'], predictions_a['predicted_labels'], average='weighted')
        model_b_f1 = f1_score(predictions_b['true_labels'], predictions_b['predicted_labels'], average='weighted')
        if model_a_f1 > model_b_f1 and mcnemar_p_value < alpha:
            print(f'Model A performs significantly better than model B by a McNemar test, p = {mcnemar_p_value}\n')
            print_per_relation_differences(predictions_a, predictions_b)
            print_classification_differences(predictions_a, predictions_b)
            #plot_per_relation_f1_differences(predictions_a, predictions_b, title=f'{split} + {grain}')
            #plot_change_in_f1_by_image_availability(split, grain, predictions_a, predictions_b, image_model_name)
        elif model_b_f1 > model_a_f1 and mcnemar_p_value < alpha:
            print(f'Model B performs significantly better than model A by a McNemar test, p = {mcnemar_p_value}\n')
            print_per_relation_differences(predictions_a, predictions_b)
            print_classification_differences(predictions_a, predictions_b)
            #plot_per_relation_f1_differences(predictions_a, predictions_b, title=f'{split} + {grain}')
            #plot_change_in_f1_by_image_availability(split, grain, predictions_a, predictions_b, image_model_name)
        else:
            print(f'There is no significant difference in performance between model A and model B, McNemar p = {mcnemar_p_value}.\n')
            print_per_relation_differences(predictions_a, predictions_b)
            print_classification_differences(predictions_a, predictions_b)

if __name__ == '__main__':
    main()