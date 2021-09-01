import pandas as pd
from os import path
import operator
import random


class MajorityClassifier(object):
    def __init__(self, mode):
        # mode determines whether the baseline will be the majority
        # class for modifiers, heads, or the overall majority class.
        assert mode in ['mod', 'head', 'overall']
        self.majority_class = None
        self.mode = mode
        self.mod_label_dict = None
        self.head_label_dict = None
        self.unique_labels = None

    def train(self, y, X_mod=None, X_head=None):
        self.unique_labels = pd.unique(y)
        y_list = list(y)
        self.majority_class = max(set(y_list), key=y_list.count)
        if self.mode == 'mod':
            assert X_mod is not None, 'If you want to run the majority modifier class baseline, ' \
                                      'you must provide the X_mod argument.'
            assert len(X_mod) == len(y), 'X_mod and y must be of the same length!'
            # dictionary structure: { 'food' : 'containment', 'bowl' : containment, ...}
            unique_mods = pd.unique(X_mod)
            mod_label_counts = {mod : {label : 0 for label in self.unique_labels} for mod in unique_mods}
            for mod, label in zip(list(X_mod), list(y)):
                mod_label_counts[mod][label] += 1
            self.mod_label_dict = \
                {mod : max(mod_label_counts[mod].items(), key=operator.itemgetter(1))[0]
                 for mod in mod_label_counts.keys()}

        elif self.mode == 'head':
            assert X_head is not None, 'If you want to run the majority head class baseline, ' \
                                       'you must provide the X_head argument.'
            assert len(X_head) == len(y), 'X_head and y must be of the same length!'
            unique_heads = pd.unique(X_head)
            head_label_counts = {head: {label: 0 for label in self.unique_labels} for head in unique_heads}
            for head, label in zip(list(X_head), list(y)):
                head_label_counts[head][label] += 1
            self.head_label_dict = \
                {head: max(head_label_counts[head].items(), key=operator.itemgetter(1))[0]
                 for head in head_label_counts.keys()}


    def classify(self, test_data):
        if self.mode == 'overall':
            if self.majority_class is None:
                raise ValueError('self.majority_class is None. Train the classifier before you try to classify!')
            return [self.majority_class]*len(test_data)
        elif self.mode == 'mod':
            num_mods_not_in_train = 0
            predictions = []
            for mod in list(test_data):
                if mod in self.mod_label_dict:
                    predictions.append(self.mod_label_dict[mod])
                else:
                    num_mods_not_in_train += 1
                    predictions.append(self.majority_class)
            print(f'NUMBER OF MODS NOT IN TRAIN: {num_mods_not_in_train}')
            return predictions
        elif self.mode == 'head':
            predictions = []
            for head in list(test_data):
                if head in self.head_label_dict:
                    predictions.append(self.head_label_dict[head])
                else:
                    predictions.append(self.majority_class)
            return predictions