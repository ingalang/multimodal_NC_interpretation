from os import path
import pandas as pd
import numpy as np
import copy
import json
from sklearn.linear_model import LogisticRegression
from itertools import product

from sparsesvd import sparsesvd
from dissect.src.composes.composition.full_additive import FullAdditive

def get_embeddings(words: list, model):
    embeddings = []
    not_in_model = []
    for word in words:
        if len(word.split()) == 1:
            if word in model:
                embeddings.append(model[word])
            else:
                embeddings.append(np.zeros(300))
                not_in_model.append(word)
        else:
            one_word_compound = ''.join(word.split())
            underscore_compound = '_'.join(word.split())
            if one_word_compound in model:
                embeddings.append(model[one_word_compound])
            elif underscore_compound in model:
                embeddings.append(model[underscore_compound])
            else:
                embeddings.append(np.zeros(300))
                not_in_model.append(word)
    return embeddings, not_in_model


def get_embedding(word, model, return_zeros=False):
    one_word = ''.join(word.split())
    underscore_compound = '_'.join(word.split())
    hyphenated_compound = '-'.join(word.split())
    if word in model:
        return model[word]
    elif hyphenated_compound in model:
        return model[hyphenated_compound]
    elif one_word in model:
        return model[one_word]
    elif underscore_compound in model:
        return model[underscore_compound]
    else:
        if return_zeros:
            return np.zeros(300)
        else: raise KeyError('Compound not found in model')


def build_semantic_space(embeddings, words):
    from dissect.src.composes.semantic_space.space import Space
    from dissect.src.composes.matrix.dense_matrix import DenseMatrix
    assert (len(embeddings) == len(words)), "Length of embedding matrix must be same as length of word list"

    emb_mat = np.matrix(embeddings)
    dense_mat = DenseMatrix(emb_mat)
    print(dense_mat.shape)

    column_names = []
    for i in range(len(embeddings[0])):
        column_names.append(f'dim_{i}')
    new_space = Space(dense_mat, words, column_names)
    return new_space


def compose_vector(mat_a, mat_b, mod_vec, head_vec):
    comp_a = np.matmul(mat_a, np.matrix(mod_vec).transpose())
    comp_b = np.matmul(mat_b, np.matrix(head_vec).transpose())
    comp_vec = comp_a + comp_b
    comp_vec = np.array(comp_vec.flatten())
    return comp_vec[0]


def get_fulladd_composed_vectors(mods, heads, mod_vectors, head_vectors, comp_model, mode):
    mat_a = comp_model._mat_a_t.get_mat()
    mat_b = comp_model._mat_b_t.get_mat()

    composed_vectors = []
    assert(len(mods) == len(heads) == len(mod_vectors) == len(head_vectors)), \
        f'Length of mods, heads, mod_vectors and head_vectors are different, ' \
        f'at respectively {len(mods), len(heads), len(mod_vectors), len(head_vectors)}'
    for i in range(len(mods)):
        mod_vec = mod_vectors[i]
        head_vec = head_vectors[i]
        mod = mods[i]
        head = heads[i]
        if mode == 'unimodal':
            comp_a = np.matmul(mat_a, np.matrix(mod_vec).transpose())
            comp_b = np.matmul(mat_b, np.matrix(head_vec).transpose())
            comp_vec = comp_a + comp_b
            comp_vec = np.array(comp_vec.flatten())
            composed_vectors.append(comp_vec[0])
        if mode=='vec_b4_concat_all':

            if mod in secondary_model:
                mod_img_vec = np.array(secondary_model[mod])
                mod_vec = mod_vec + mod_img_vec
            if head in secondary_model:
                head_img_vec = np.array(secondary_model[head])
                head_vec = head_vec + head_img_vec
            #TODO fix this matrix thing
            comp_a = np.matmul(mat_a, np.matrix(mod_vec).transpose())
            comp_b = np.matmul(mat_b, np.matrix(head_vec).transpose())
            comp_vec = comp_a + comp_b
            comp_vec = np.array(comp_vec.flatten())
            composed_vectors.append(comp_vec[0])
        if mode=='vec_after_concat':
            compound = mod + ' ' + head
    return composed_vectors




def main(SPLIT, GRAIN):
    # configurations
    REMOVE_NON_BINARY = True
    REMOVE_NON_COMP = True
    GLOVE = False

    POSSIBLE_MODES = ['vec_b4_concat_all', 'unimodal', 'vec_after_concat']
    MODE = 'unimodal'
    assert MODE in POSSIBLE_MODES, \
        f'Parameter MODE must be one of the following strings: {POSSIBLE_MODES}'

    POSSIBLE_MODEL_NAMES = ['images', 'word2vec', 'glove']
    # 'images', 'glove', or 'word2vec'
    MODEL_NAME = 'word2vec'

    SECONDARY_MODEL_NAME = 'images'
    FILTER_COMPOSITION_TRAINING_ON_SECONDARY_MODEL = False

    global model, secondary_model
    main_model = get_model(MODEL_NAME)

    if FILTER_COMPOSITION_TRAINING_ON_SECONDARY_MODEL:
        assert(SECONDARY_MODEL_NAME in POSSIBLE_MODEL_NAMES), \
            f'Please supply a secondary model name from this list: {POSSIBLE_MODEL_NAMES}'


    if MODE != 'unimodal':
        assert(SECONDARY_MODEL_NAME), 'If you are running a multimodal experiment, you must input a secondary model.'
    secondary_model = get_model(SECONDARY_MODEL_NAME)

    data_path = f'../data/Tratz_2011_data_comp_binary/tratz2011_cb_{GRAIN}_grained_{SPLIT}'

    train_data = load_data(data_path, 'train')
    test_data = load_data(data_path, 'test')
    val_data = load_data(data_path, 'val')

    ##Filtering data on text model
    train_data = remove_non_existing(train_data, main_model, filter_on=['mod', 'head'])
    test_data = remove_non_existing(test_data, main_model, filter_on=['mod', 'head'])
    val_data = remove_non_existing(val_data, main_model, filter_on=['mod', 'head'])

    ##FILTERING DATA ON IMAGE MODEL
    train_data = remove_non_existing(train_data, secondary_model, filter_on=['mod', 'head'])
    test_data = remove_non_existing(test_data, secondary_model, filter_on=['mod', 'head'])
    #print(f'LENGTH OF TRAIN AFTER FILTERING: {len(train_data)}')
    #print(f'LENGTH OF TEST AFTER FILTERING: {len(test_data)}')
    val_data = remove_non_existing(val_data, secondary_model, filter_on=['mod', 'head'])

    print("Length of train_data: {}".format(len(train_data)))
    print("Length of test_data: {}".format(len(test_data)))
    print("Length of val_data: {}".format(len(val_data)))

    train_data_exists_in_model = remove_non_existing(train_data, model=main_model, filter_on=['compound'])
    train_data_exists_in_model = remove_non_existing(train_data_exists_in_model, model=secondary_model, filter_on=['compound'])
    print(train_data_exists_in_model)
    new_train = train_data_exists_in_model.drop_duplicates(subset=['nc_mod'])
    new_train = new_train.drop_duplicates(subset=['nc_head'])
    new_train = new_train.drop_duplicates(subset=['compound'])

    space_train_mod_embeddings, train_not_in_model = get_embeddings(new_train['nc_mod'], main_model)
    space_train_head_embeddings, _ = get_embeddings(new_train['nc_head'], main_model)
    space_train_compound_embeddings, _ = get_embeddings(new_train['compound'], main_model)

    assert (len(space_train_mod_embeddings) == len(space_train_head_embeddings) == len(
        space_train_compound_embeddings)), 'size of spaces must be the same'
    TRAIN_SUPPORT = len(space_train_mod_embeddings)

    mod_space = build_semantic_space(space_train_mod_embeddings, list(new_train['nc_mod']))
    head_space = build_semantic_space(space_train_head_embeddings, list(new_train['nc_head']))
    compound_space = build_semantic_space(space_train_compound_embeddings, list(new_train['compound']))

    # Training the fullAdditive composition model

    comp_train_data = [(mod, head, compound)
                       for mod, head, compound
                       in zip(new_train['nc_mod'], new_train['nc_head'], new_train['compound'])]

    comp_model = FullAdditive()
    comp_model._regression_learner._intercept = False
    comp_model.train(comp_train_data, (mod_space, head_space), compound_space)

    train_mod_embeddings, _ = get_embeddings(train_data['nc_mod'], main_model)
    train_head_embeddings, _ = get_embeddings(train_data['nc_head'], main_model)
    #print(len(train_mod_embeddings), len(train_head_embeddings))

    test_mod_embeddings, _ = get_embeddings(test_data['nc_mod'], main_model)
    test_head_embeddings, _ = get_embeddings(test_data['nc_head'], main_model)
    #print(len(test_mod_embeddings), len(test_head_embeddings))

    val_mod_embeddings, _ = get_embeddings(val_data['nc_mod'], main_model)
    val_head_embeddings, _ = get_embeddings(val_data['nc_head'], main_model)
    #print(len(val_mod_embeddings), len(val_head_embeddings))

    train_labels = train_data['nc_type']
    test_labels = test_data['nc_type']
    val_labels = val_data['nc_type']

    train_features = get_fulladd_composed_vectors(list(train_data['nc_mod']), list(train_data['nc_head']), train_mod_embeddings, train_head_embeddings, comp_model, mode=MODE)
    test_features = get_fulladd_composed_vectors(list(test_data['nc_mod']), list(test_data['nc_head']), test_mod_embeddings, test_head_embeddings, comp_model, mode=MODE)
    val_features = get_fulladd_composed_vectors(list(val_data['nc_mod']), list(val_data['nc_head']), val_mod_embeddings, val_head_embeddings, comp_model, mode=MODE)

    print(len(train_features), len(train_labels))
    #print(len(val_features), len(val_labels))

    # val_preds = classify(train_features, train_labels, val_features)
    # _, _, _, val_f1, val_acc, val_full_report = evaluate(val_labels, val_preds)
    # print('{0:*^80}'.format(' VALIDATION REPORT '))
    # print(val_full_report)

    test_preds = classify(train_features, train_labels, test_features)
    _, _, _, test_f1, test_acc, test_full_report = evaluate(test_labels, test_preds)
    print('{0:*^80}'.format(' TEST REPORT '))
    print(test_full_report)

    print(f'Train support: {TRAIN_SUPPORT}')
    print(SPLIT, GRAIN)



############## ALL THE SAVING AND STUFF CAN BE IMPLEMENTED LATER ############
    # update_csv_with_results(val_acc, val_f1, test_acc, test_f1,
    #                         glove=GLOVE, split=SPLIT, grain=GRAIN,
    #                         remove_non_binary=REMOVE_NON_BINARY,
    #                         remove_non_comp=REMOVE_NON_COMP, train_support=TRAIN_SUPPORT)
    #
    # if not GLOVE:
    #     save_predictions(f'fulladd-val-preds_{SPLIT}_{GRAIN}_{REMOVE_NON_BINARY}_{REMOVE_NON_COMP}', val_data,
    #                      val_preds)
    #     save_predictions(f'fulladd-test-preds_{SPLIT}_{GRAIN}_{REMOVE_NON_BINARY}_{REMOVE_NON_COMP}', test_data,
    #                      test_preds)
    # else:
    #     save_predictions(f'glove-fulladd-val-preds_{SPLIT}_{GRAIN}_{REMOVE_NON_BINARY}_{REMOVE_NON_COMP}', val_data,
    #                      val_preds)
    #     save_predictions(f'glove-fulladd-test-preds_{SPLIT}_{GRAIN}_{REMOVE_NON_BINARY}_{REMOVE_NON_COMP}', test_data,
    #                      test_preds)


if __name__ == '__main__':
    splits = ['random', 'lexical_full','lexical_mod', 'lexical_head']
    grains = ['coarse', 'fine']

    for combo in product(splits, grains):
        main(combo[0], combo[1])

