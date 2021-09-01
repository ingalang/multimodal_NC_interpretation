import argparse
from models.utils import load_data, remove_non_existing, get_model, combine_vectors, save_predictions
from training.compositional.train_fulladd_matrices import train_fulladd_matrices
from models.full_add_composition import get_embeddings, compose_vector
from models.multimodal_composition import get_multimodal_fulladd_vectors
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
import pandas as pd
from os import path
from models.baseline import MajorityClassifier
from itertools import product
from sklearn.decomposition import PCA
import numpy as np


def evaluate(labels, predictions):
    cls_report = metrics.classification_report(labels, predictions, digits=3, zero_division=0)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return precision, recall, support, f1, accuracy, cls_report

def get_classifier():
    classifier = LinearSVC(penalty='l2', C=0.5, max_iter=10000)
    return classifier


# TODO denne skal kanskje vekk, eller?
def update_csv_with_results(val_acc,
                            val_f1,
                            test_acc,
                            test_f1,
                            glove,
                            split,
                            grain,
                            remove_non_binary,
                            remove_non_comp,
                            train_support
                            ):
    if glove:
        model_name = 'unimodal-fulladd-svm-glove'
    else:
        model_name = 'unimodal-fulladd-svm'

    new_results = {
        'model_name': model_name,
        'split': split,
        'grain': grain,
        'non_binary_removed': remove_non_binary,
        'non_comp_removed': remove_non_comp,
        'val_acc': val_acc,
        'val_f1_weighted': val_f1,
        'test_acc': test_acc,
        'test_f1_weighted': test_f1,
        'train_support': train_support
    }

    csv_filename = 'results/composition_fulladd_glove_svm_unimodal_auto.csv'
    if path.exists(csv_filename):
        results = pd.read_csv(csv_filename)
        results = results.append(new_results, ignore_index=True)
    else:
        results = pd.DataFrame(new_results, index=[0])

    print(results)
    results.to_csv(csv_filename, index=False)
    print("SAVED RESULTS")

def run_classification(split: str,
                       grain: str,
                       modality: str,
                       mode: str,
                       math_mode:str,
                       primary_model,
                       secondary_model=None,
                       primary_model_name=None,
                       secondary_model_name=None,
                       print_style='simple',
                       filter_data_by=None,
                       save_preds=True):
    if filter_data_by is None:
        filter_data_by = []
    if modality=='multimodal' and isinstance(secondary_model, type(None)):
        raise ValueError('To run a multimodal experiment, you must input both a text model and an image model')

    data_path = f'../data/Tratz_2011_data_comp_binary/tratz2011_cb_{grain}_grained_{split}'

    train_data = load_data(data_path, 'train')
    test_data = load_data(data_path, 'test')
    val_data = load_data(data_path, 'val')
    print(f'Length of train after loading: {len(train_data)}')
    print(f'Length of test after loading: {len(test_data)}')
    print(f'Length of val after loading: {len(val_data)}\n')

    for model in filter_data_by:
        train_data = remove_non_existing(train_data, model, filter_on=['mod', 'head'])
        test_data = remove_non_existing(test_data, model, filter_on=['mod', 'head'])
        val_data = remove_non_existing(val_data, model, filter_on=['mod', 'head'])
        print(f'Length of train after filtering on mod and head: {len(train_data)}')
        print(f'Length of test after filtering on mod and head: {len(test_data)}')
        print(f'Length of val after filtering on mod and head: {len(val_data)} \n')

    # getting labels for train, test, and val
    train_labels = train_data['nc_type']
    test_labels = test_data['nc_type']
    val_labels = val_data['nc_type']

    if modality=='baseline':
        cls_overall = MajorityClassifier(mode='overall')
        cls_mod = MajorityClassifier(mode='mod')
        cls_head = MajorityClassifier(mode='head')

        cls_overall.train(y=train_data['nc_type'])
        cls_mod.train(y=train_data['nc_type'], X_mod=train_data['nc_mod'])
        cls_head.train(y=train_data['nc_type'], X_head=train_data['nc_head'])

        overall_preds = cls_overall.classify(test_data['nc_mod'])
        mod_preds = cls_mod.classify(test_data['nc_mod'])
        head_preds = cls_head.classify(test_data['nc_head'])

        _, _, _, f1_overall, accuracy_overall, cls_report_overall = evaluate(list(test_data['nc_type']), overall_preds)
        _, _, _, f1_mod, accuracy_mod, cls_report_mod = evaluate(list(test_data['nc_type']), mod_preds)
        _, _, _, f1_head, accuracy_head, cls_report_head = evaluate(list(test_data['nc_type']), head_preds)

        print(f'BASELINE RESULTS FOR TEST DATA ON {split} + {grain}: \n')
        if print_style=='simple':
            print('Overall majority classifier:')
            print(f'Weighted F1: {f1_overall}')
            print(f'Accuracy: {accuracy_overall} \n')

            print('Modifier majority classifier:')
            print(f'Weighted F1: {f1_mod}')
            print(f'Accuracy: {accuracy_mod} \n')

            print('Head majority classifier:')
            print(f'Weighted F1: {f1_head}')
            print(f'Accuracy: {accuracy_head} \n')
        elif print_style=='full':
            print('Overall majority classifier:')
            print(cls_report_overall)
            print()

            print('Modifier majority classifier:')
            print(cls_report_mod)
            print()

            print('Head majority classifier:')
            print(cls_report_head)
            print()

        if save_preds:

            print('Saving predictions...')

            save_predictions(
                data=test_data,
                preds=overall_preds,
                split=split,
                grain=grain,
                modality=modality + '_overall',
                mode=mode,
                math_mode=math_mode,
                primary_model_name=primary_model_name,
                secondary_model_name=secondary_model_name,
                filtered_on=len(filter_data_by)
            )

            save_predictions(
                data=test_data,
                preds=head_preds,
                split=split,
                grain=grain,
                modality=modality + '_head',
                mode=mode,
                math_mode=math_mode,
                primary_model_name=primary_model_name,
                secondary_model_name=secondary_model_name,
                filtered_on=len(filter_data_by)
            )

            save_predictions(
                data=test_data,
                preds=mod_preds,
                split=split,
                grain=grain,
                modality=modality + '_mod',
                mode=mode,
                math_mode=math_mode,
                primary_model_name=primary_model_name,
                secondary_model_name=secondary_model_name,
                filtered_on=len(filter_data_by)
            )

    elif modality=='unimodal_text' or modality=='unimodal_images':
        # getting embeddings for train mods and heads
        train_mod_embeddings, _ = get_embeddings(train_data['nc_mod'], primary_model)
        train_head_embeddings, _ = get_embeddings(train_data['nc_head'], primary_model)

        # getting embeddings for test mods and heads
        test_mod_embeddings, _ = get_embeddings(test_data['nc_mod'], primary_model)
        test_head_embeddings, _ = get_embeddings(test_data['nc_head'], primary_model)

        # getting embeddings for val mods and heads
        val_mod_embeddings, _ = get_embeddings(val_data['nc_mod'], primary_model)
        val_head_embeddings, _ = get_embeddings(val_data['nc_head'], primary_model)


        if math_mode == 'fulladd':
            if len(filter_data_by) >= 2:
                second_model = filter_data_by[1]
                if len(filter_data_by) == 3:
                    train_data = remove_non_existing(train_data, filter_data_by[2], filter_on=['mod', 'head', 'compound'])
            else:
                second_model = None

            mat_a, mat_b, train_support = train_fulladd_matrices(train_data, primary_model, second_model)
            train_features = [compose_vector(mat_a, mat_b, mod_vec, head_vec)
                              for mod_vec, head_vec in zip(train_mod_embeddings, train_head_embeddings)]
            test_features = [compose_vector(mat_a, mat_b, mod_vec, head_vec)
                              for mod_vec, head_vec in zip(test_mod_embeddings, test_head_embeddings)]
            val_features = [compose_vector(mat_a, mat_b, mod_vec, head_vec)
                              for mod_vec, head_vec in zip(val_mod_embeddings, val_head_embeddings)]
            print(f'TRAIN SUPPORT: {train_support}')
        elif math_mode == 'multiply' or math_mode == 'concatenate':
            train_features = [combine_vectors(mod_vec, head_vec, math_mode)
                              for mod_vec, head_vec in zip(train_mod_embeddings, train_head_embeddings)]
            test_features = [combine_vectors(mod_vec, head_vec, math_mode)
                              for mod_vec, head_vec in zip(test_mod_embeddings, test_head_embeddings)]
            val_features = [combine_vectors(mod_vec, head_vec, math_mode)
                              for mod_vec, head_vec in zip(val_mod_embeddings, val_head_embeddings)]
        else:
            raise ValueError('Invalid mode. Must be one of `fulladd`, `multiply`, `concatenate`.')

        classifier = get_classifier()
        classifier.fit(train_features, train_labels)
        test_preds = classifier.predict(test_features)
        val_preds = classifier.predict(val_features)

        _, _, _, test_f1, test_accuracy, test_report = evaluate(test_labels, test_preds)
        _, _, _, val_f1, val_accuracy, val_report = evaluate(val_labels, val_preds)

        print(f'{modality} RESULTS FOR {split} + {grain} + {math_mode} + {primary_model_name} FILTERED ON {len(filter_data_by)} MODELS:')
        if print_style == 'simple':
            print(f'Weighted F1, val: {val_f1}')
            print(f'Accuracy, val: {val_accuracy}\n')
            print(f'Weighted F1, test: {test_f1}')
            print(f'Accuracy, test: {test_accuracy}\n')
        if print_style == 'full':
            print('VAL RESULTS: ')
            print(val_report)
            print()

            print('TEST RESULTS: ')
            print(test_report)
            print()
        if save_preds:
            print('Saving predictions...')
            save_predictions(
                data=test_data,
                preds=test_preds,
                split=split,
                grain=grain,
                modality=modality,
                mode=mode,
                math_mode=math_mode,
                primary_model_name=primary_model_name,
                secondary_model_name=secondary_model_name,
                filtered_on=len(filter_data_by)
            )

    elif modality=='multimodal':
        ### TEXT EMBEDDINGS
        # getting text embeddings for train mods and heads
        train_mod_embeddings_text, _ = get_embeddings(train_data['nc_mod'], primary_model)
        train_head_embeddings_text, _ = get_embeddings(train_data['nc_head'], primary_model)

        # getting text embeddings for test mods and heads
        test_mod_embeddings_text, _ = get_embeddings(test_data['nc_mod'], primary_model)
        test_head_embeddings_text, _ = get_embeddings(test_data['nc_head'], primary_model)

        # getting text embeddings for val mods and heads
        val_mod_embeddings_text, _ = get_embeddings(val_data['nc_mod'], primary_model)
        val_head_embeddings_text, _ = get_embeddings(val_data['nc_head'], primary_model)

        ### IMAGE EMBEDDINGS
        # getting image embeddings for train mods and heads
        train_mod_embeddings_images, _ = get_embeddings(train_data['nc_mod'], secondary_model)
        train_head_embeddings_images, _ = get_embeddings(train_data['nc_head'], secondary_model)

        # getting image embeddings for test mods and heads
        test_mod_embeddings_images, _ = get_embeddings(test_data['nc_mod'], secondary_model)
        test_head_embeddings_images, _ = get_embeddings(test_data['nc_head'], secondary_model)

        # getting image embeddings for val mods and heads
        val_mod_embeddings_images, _ = get_embeddings(val_data['nc_mod'], secondary_model)
        val_head_embeddings_images, _ = get_embeddings(val_data['nc_head'], secondary_model)

        if math_mode=='fulladd':
            if len(filter_data_by) == 3:
                third_model = filter_data_by[2]
            else:
                third_model = None
            train_features, test_features, val_features = get_multimodal_fulladd_vectors(train_data=train_data,
                                                                                         test_data=test_data,
                                                                                         val_data=val_data,
                                                                                         text_model=primary_model,
                                                                                         image_model=secondary_model,
                                                                                         order_of_composition=mode,
                                                                                         third_model=third_model)
            print('length of features ', len(train_features), len(test_features), len(val_features))
        elif math_mode == 'concatenate_mod':
            train_multimodal_mod_vectors = [combine_vectors(mod_text_vector, mod_image_vector, 'concatenate')
                                            for mod_text_vector, mod_image_vector
                                            in zip(train_mod_embeddings_text, train_mod_embeddings_images)]
            test_multimodal_mod_vectors = [combine_vectors(mod_text_vector, mod_image_vector, 'concatenate')
                                           for mod_text_vector, mod_image_vector
                                           in zip(test_mod_embeddings_text, test_mod_embeddings_images)]
            val_multimodal_mod_vectors = [combine_vectors(mod_text_vector, mod_image_vector, 'concatenate')
                                          for mod_text_vector, mod_image_vector
                                          in zip(val_mod_embeddings_text, val_mod_embeddings_images)]

            train_features = [combine_vectors(multimodal_mod_vector, head_vector, 'concatenate')
                              for multimodal_mod_vector, head_vector
                              in zip(train_multimodal_mod_vectors, train_head_embeddings_text)]
            test_features = [combine_vectors(multimodal_mod_vector, head_vector, 'concatenate')
                              for multimodal_mod_vector, head_vector
                              in zip(test_multimodal_mod_vectors, test_head_embeddings_text)]
            val_features = [combine_vectors(multimodal_mod_vector, head_vector, 'concatenate')
                              for multimodal_mod_vector, head_vector
                              in zip(val_multimodal_mod_vectors, val_head_embeddings_text)]
        elif math_mode == 'concatenate_head':

            train_multimodal_head_vectors = [combine_vectors(head_text_vector, head_image_vector, 'concatenate')
                                            for head_text_vector, head_image_vector
                                            in zip(train_head_embeddings_text, train_head_embeddings_images)]
            test_multimodal_head_vectors = [combine_vectors(head_text_vector, head_image_vector, 'concatenate')
                                             for head_text_vector, head_image_vector
                                             in zip(test_head_embeddings_text, test_head_embeddings_images)]
            val_multimodal_head_vectors = [combine_vectors(head_text_vector, head_image_vector, 'concatenate')
                                             for head_text_vector, head_image_vector
                                             in zip(val_head_embeddings_text, val_head_embeddings_images)]

            train_features = [combine_vectors(mod_vector, multimodal_head_vector, 'concatenate')
                              for mod_vector, multimodal_head_vector
                              in zip(train_mod_embeddings_text, train_multimodal_head_vectors)]
            test_features = [combine_vectors(mod_vector, multimodal_head_vector, 'concatenate')
                              for mod_vector, multimodal_head_vector
                              in zip(test_mod_embeddings_text, test_multimodal_head_vectors)]
            val_features = [combine_vectors(mod_vector, multimodal_head_vector, 'concatenate')
                              for mod_vector, multimodal_head_vector
                              in zip(val_mod_embeddings_text, val_multimodal_head_vectors)]

        else:
            #if math_mode is 'multiply' or 'concatenate'
            if mode=='similar_first':
                ### COMBINING MOD & HEAD VECTORS OF THE SAME MODALITY FIRST
                # combining mod + head text vectors
                train_compound_vectors_text = [combine_vectors(mod_vector, head_vector, math_mode)
                                               for mod_vector, head_vector
                                               in zip(train_mod_embeddings_text, train_head_embeddings_text)]
                test_compound_vectors_text = [combine_vectors(mod_vector, head_vector, math_mode)
                                               for mod_vector, head_vector
                                               in zip(test_mod_embeddings_text, test_head_embeddings_text)]
                val_compound_vectors_text = [combine_vectors(mod_vector, head_vector, math_mode)
                                              for mod_vector, head_vector
                                              in zip(val_mod_embeddings_text, val_head_embeddings_text)]

                # combining mod & head image vectors
                train_compound_vectors_images = [combine_vectors(mod_vector, head_vector, math_mode)
                                               for mod_vector, head_vector
                                               in zip(train_mod_embeddings_images, train_head_embeddings_images)]
                test_compound_vectors_images = [combine_vectors(mod_vector, head_vector, math_mode)
                                              for mod_vector, head_vector
                                              in zip(test_mod_embeddings_images, test_head_embeddings_images)]
                val_compound_vectors_images = [combine_vectors(mod_vector, head_vector, math_mode)
                                             for mod_vector, head_vector
                                             in zip(val_mod_embeddings_images, val_head_embeddings_images)]

                ### THEN COMBINING THE RESULTING VECTORS INTO THE FINAL FEATURE VECTORS
                train_features = [combine_vectors(text_vector, image_vector, math_mode)
                                  for text_vector, image_vector
                                  in zip(train_compound_vectors_text, train_compound_vectors_images)]
                test_features = [combine_vectors(text_vector, image_vector, math_mode)
                                  for text_vector, image_vector
                                  in zip(test_compound_vectors_text, test_compound_vectors_images)]
                val_features = [combine_vectors(text_vector, image_vector, math_mode)
                                  for text_vector, image_vector
                                  in zip(val_compound_vectors_text, val_compound_vectors_images)]
            elif mode=='opposite_first':
                ### COMBINING MOD TEXT VECTORS WITH MOD IMAGE VECTORS; AND SIMILARLY WITH HEAD VECTORS
                # mod vectors: text and images
                train_multimodal_mod_vectors = [combine_vectors(mod_text_vector, mod_image_vector, math_mode)
                                                for mod_text_vector, mod_image_vector
                                                in zip(train_mod_embeddings_text, train_mod_embeddings_images)]

                test_multimodal_mod_vectors = [combine_vectors(mod_text_vector, mod_image_vector, math_mode)
                                                for mod_text_vector, mod_image_vector
                                                in zip(test_mod_embeddings_text, test_mod_embeddings_images)]
                val_multimodal_mod_vectors = [combine_vectors(mod_text_vector, mod_image_vector, math_mode)
                                                for mod_text_vector, mod_image_vector
                                                in zip(val_mod_embeddings_text, val_mod_embeddings_images)]

                # head vectors: text and images
                train_multimodal_head_vectors = [combine_vectors(head_text_vector, head_image_vector, math_mode)
                                                for head_text_vector, head_image_vector
                                                in zip(train_head_embeddings_text, train_head_embeddings_images)]

                test_multimodal_head_vectors = [combine_vectors(head_text_vector, head_image_vector, math_mode)
                                               for head_text_vector, head_image_vector
                                               in zip(test_head_embeddings_text, test_head_embeddings_images)]
                val_multimodal_head_vectors = [combine_vectors(head_text_vector, head_image_vector, math_mode)
                                              for head_text_vector, head_image_vector
                                              in zip(val_head_embeddings_text, val_head_embeddings_images)]

                ### THEN COMBINING THE RESULTING MULTIMODAL MOD VECTORS AND MULTIMODAL HEAD VECTORS
                train_features = [combine_vectors(mod_vector, head_vector, math_mode)
                                          for mod_vector, head_vector
                                          in zip(train_multimodal_mod_vectors, train_multimodal_head_vectors)]
                test_features = [combine_vectors(mod_vector, head_vector, math_mode)
                                          for mod_vector, head_vector
                                          in zip(test_multimodal_mod_vectors, test_multimodal_head_vectors)]
                val_features = [combine_vectors(mod_vector, head_vector, math_mode)
                                          for mod_vector, head_vector
                                          in zip(val_multimodal_mod_vectors, val_multimodal_head_vectors)]
            else:
                raise ValueError('Invalid mode. Use `similar_first` or `opposite_first`.')

        classifier = get_classifier()
        print('shape of trainfeatures: ', (np.array(train_features)).shape)
        classifier.fit(train_features, train_labels)
        test_preds = classifier.predict(test_features)
        val_preds = classifier.predict(val_features)

        _, _, _, test_f1, test_accuracy, test_report = evaluate(test_labels, test_preds)
        _, _, _, val_f1, val_accuracy, val_report = evaluate(val_labels, val_preds)

        print(f'{modality} RESULTS FOR {split} + {grain} + {math_mode} + {mode} + {primary_model_name} FILTERED ON {len(filter_data_by)} MODELS:')
        if print_style == 'simple':
            print(f'Weighted F1, val: {val_f1}')
            print(f'Accuracy, val: {val_accuracy}\n')
            print(f'Weighted F1, test: {test_f1}')
            print(f'Accuracy, test: {test_accuracy}\n')
        if print_style == 'full':
            print('VAL RESULTS: ')
            print(val_report)
            print()

            print('TEST RESULTS: ')
            print(test_report)
            print()

        if save_preds:
            save_predictions(
                data=test_data,
                preds=test_preds,
                split=split,
                grain=grain,
                modality=modality,
                mode=mode,
                math_mode=math_mode,
                primary_model_name=primary_model_name,
                secondary_model_name=secondary_model_name,
                filtered_on=len(filter_data_by)
            )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=False, default='all',
                        choices=['random', 'lexical_full', 'lexical_head', 'lexical_mod', 'all'],
                        help='What data split to use (random, lexical_full, lexical_head, lexical_mod, all)')
    parser.add_argument('--grain', type=str, required=False, default='all',
                        choices=['coarse', 'fine', 'all'],
                        help='What grain (fineness of relation classes) to use '
                             '(fine, with 37 classes; coarse, with 12 classes; or all, which runs both)')
    parser.add_argument('--modality', type=str, required=True,
                        choices=['unimodal_text', 'unimodal_images', 'multimodal', 'baseline'],
                        help='Which modality to use (unimodal_text, unimodal_images, multimodal, or baseline)')
    parser.add_argument('--mode', type=str, required=False, default='similar_first',
                        choices=['similar_first', 'opposite_first'],
                        help='Which mode to use for training and combining the multimodal vectors. \n'
                             'similar_first: composes textual and image vectors separately and then combines them. \n'
                             'opposite_first: combines textual mod vectors with image mod vectors, '
                             'and similarly for head vectors. Then combines the resulting multimodal mod & head vectors ')
    parser.add_argument('--math_mode', type=str, required=False, default='concatenate',
                        choices=['fulladd', 'multiply', 'concatenate', 'concatenate_mod', 'concatenate_head'],
                        help='Math mode to use when combining vectors together. \n'
                             'Options: fulladd, multiply, concatenate \n'
                             'Default: fulladd')
    parser.add_argument('--text_model', type=str, required=False, default='word2vec',
                        choices=['word2vec', 'glove', 'imagined'],
                        help='What pre-trained model to use for textual embeddings. \n'
                             'Default: word2vec')
    parser.add_argument('--image_model', type=str, required=False, default='ResNet_10_norm',
                        choices=['ResNet_10', 'ResNet_100', 'ResNet_10_norm',
                                 'ResNet_100_norm', 'ResNet_10_150_norm',
                                 'ResNet_norm_300', 'imagined'],
                        help='What pre-computed embeddings to use for the images. \n'
                             'Options: ResNet_10, ResNet_100, ResNet_10_norm, ResNet_100_norm, ResNet_10_150_norm, imagined \n'
                             'Default: ResNet_10 (ResNet152V2 vectors averaged over 10 images per word, '
                             'and PCA-reduced to 300 dimensions.')
    parser.add_argument('--filter_data_by_model', type=int, required=True, choices=[0, 1, 2, 3],
                        help='How to filter the data. \n'
                             'Type=int \n'
                             'Options: 0 means no filtering. 1 filters on the primary model '
                             '(which is the text model unless modality is unimodal_images). '
                             '2 filters on both the primary model and secondary model.')
    parser.add_argument('--save_preds', required=False, type=bool, default=True,
                        help='Whether to save the test predictions to a file, in order to run statistics on it later.')

    args = parser.parse_args()
    split, grain, modality, mode, math_mode, text_model_name, image_model_name, filter_data_by_model = \
        args.split, args.grain, args.modality, args.mode, args.math_mode, \
        args.text_model, args.image_model, args.filter_data_by_model

    if not(modality=='baseline' and filter_data_by_model==0):
        text_model = get_model(text_model_name)
        image_model = get_model(image_model_name)
    else:
        text_model = None
        image_model = None


    if modality=='unimodal_images':
        primary_model = image_model
        primary_model_name = image_model_name

        secondary_model = text_model
        secondary_model_name = text_model_name
    else:
        primary_model = text_model
        primary_model_name = text_model_name

        secondary_model = image_model
        secondary_model_name = image_model_name

    del text_model, image_model

    if filter_data_by_model == 0:
        filter_data_by = []
    elif filter_data_by_model == 1:
        filter_data_by = [primary_model]
    elif filter_data_by_model == 2:
        filter_data_by = [primary_model, secondary_model]
    elif filter_data_by_model == 3:
        third_model = get_model('ResNet_100')
        filter_data_by = [primary_model, secondary_model, third_model]
    else:
        raise ValueError('filter_data_by_model parameter (int) must be either 0, 1, or 2.')

    if split == 'all':
        splits = ['random', 'lexical_full', 'lexical_mod', 'lexical_head']
    else:
        splits = [split]
    if grain == 'all':
        grains = ['coarse', 'fine']
    else:
        grains = [grain]

    for combo in product(splits, grains):
        split = combo[0]
        grain = combo[1]

        run_classification(split=split,
                           grain=grain,
                           modality=modality,
                           mode=mode,
                           math_mode=math_mode,
                           primary_model=primary_model,
                           secondary_model=secondary_model,
                           filter_data_by=filter_data_by,
                           primary_model_name=primary_model_name,
                           secondary_model_name=secondary_model_name)


if __name__ == '__main__':
    main()
