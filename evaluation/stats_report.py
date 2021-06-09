import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os
from scipy import stats
import numpy as np

pd.set_option('display.max_columns', None)

# This function was adapted from Rotem Dror's repository
# https://github.com/rtmdrr/testSignificanceNLP/blob/master/testSignificance.py
def mcNemar(table, model_a_name, model_b_name):
    a_correct_b_incorrect = table.loc[f'{model_a_name} correct', f'{model_b_name} incorrect']
    a_incorrect_b_correct = table.loc[f'{model_a_name} incorrect', f'{model_b_name} correct']
    statistic = float(np.abs(a_correct_b_incorrect-a_incorrect_b_correct))**2/(a_incorrect_b_correct+a_correct_b_incorrect)
    pval = 1-stats.chi2.cdf(statistic,1)
    return pval

def make_contingency_table(preds_a: pd.DataFrame, preds_b: pd.DataFrame, model_a_name, model_b_name):
    assert (len(preds_a) == len(preds_b)), 'The two DataFrames to compare must be of same length!'
    assert (len(pd.unique(preds_a.nc_type)) == len(pd.unique(preds_b.nc_type))), \
        f'Prediction file A and prediction file B contain different numbers of labels. ' \
        f'File A contains {len(pd.unique(preds_a.nc_type))} and file B contains {len(pd.unique(preds_b.nc_type))}.'
    unique_existing_labels = pd.unique(pd.concat([preds_a.nc_type, preds_b.nc_type], axis=0))
    contingency_table = pd.DataFrame({f'{model_b_name} correct' : 0, f'{model_b_name} incorrect' : 0},
                                     index=pd.Index([f'{model_a_name} correct', f'{model_a_name} incorrect']),
                                     columns=pd.Index([f'{model_b_name} correct', f'{model_b_name} incorrect']))
    print(contingency_table)
    for i, row in preds_a.iterrows():
        sample_mod = row['nc_mod']
        sample_head = row['nc_head']
        comparable_sample_in_b = preds_b.loc[(preds_b['nc_mod'] == sample_mod) & (preds_b['nc_head'] == sample_head)]
        assert len(comparable_sample_in_b) == 1, 'Some data from preds_a does not exist in preds_b, ' \
                                                 'or preds_b contains multiples of the same data sample'
        assert (str(row['nc_type']) == str(
            comparable_sample_in_b['nc_type'].values[0])), 'Two identical data samples have different gold labels'

        model_a_prediction = row['predicted_type']
        model_b_prediction = comparable_sample_in_b['predicted_type'].values[0]
        true_label = comparable_sample_in_b['nc_type'].values[0]

        # if model A is correct...
        if model_a_prediction == true_label:
            # if model B is also correct...
            if model_b_prediction == true_label:
                contingency_table.loc[f'{model_a_name} correct', f'{model_b_name} correct'] += 1
            # if model B is incorrect...
            else:
                contingency_table.loc[f'{model_a_name} correct', f'{model_b_name} incorrect'] += 1
        # if model A is incorrect...
        else:
            # if model B is correct...
            if model_b_prediction == true_label:
                contingency_table.loc[f'{model_a_name} incorrect', f'{model_b_name} correct'] += 1
            # if model B is also incorrect...
            else:
                contingency_table.loc[f'{model_a_name} incorrect', f'{model_b_name} incorrect'] += 1
    return contingency_table

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
    pred = data.predicted_type
    all_labels = pd.concat([true, pred], axis=0)
    unique_labels = pd.unique(all_labels)
    print(unique_labels)
    print(confusion_matrix(y_true=true, y_pred=pred))

def print_classification_report(data: pd.DataFrame):
    true = data.nc_type
    pred = data.predicted_type
    all_labels = pd.concat([true, pred], axis=0)
    unique_labels = pd.unique(all_labels)
    print(unique_labels)
    print(classification_report(true, pred, labels=unique_labels, zero_division=0))

def load_pred_file(filename):
    data = pd.read_csv(filename)
    if 'predicted_id' in data.columns:
        data.rename(columns={'predicted_id': 'predicted_type'}, inplace=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_a', type=str, required=True, help='A file to evaluate')
    parser.add_argument('--preds_a_name', type=str, required=False,
                        help='Name of the model that produced the first prediction file', default='MODEL_A')

    #todo skal preds_b være required?
    parser.add_argument('--preds_b', type=str, required=True, help='Another file to compare with the first file')
    parser.add_argument('--preds_b_name', type=str, required=False,
                        help='Name of the model that produced the second prediction file', default='MODEL_B')

    args = parser.parse_args()
    filename_a, filename_b, model_a_name, model_b_name = args.preds_a, args.preds_b, args.preds_a_name, args.preds_b_name

    assert(filename_a.endswith('.csv')), f'{filename_a} is not a csv file! Only csv files are accepted.'
    assert(filename_b.endswith('.csv')), f'{filename_b} is not a csv file! Only csv files are accepted.'


    data_a = load_pred_file(filename_a)
    data_b = load_pred_file(filename_b)

    print(len(data_a))
    metadata_a = get_file_metadata(filename_a)
    metadata_b = get_file_metadata(filename_b)

    print_single_confusion_matrix(data_a)

    print(metadata_a)
    print(metadata_b)
    contingency_table = make_contingency_table(preds_a=data_a, preds_b=data_b, model_a_name=model_a_name, model_b_name=model_b_name)
    print(contingency_table)

    mcnemar_p = mcNemar(contingency_table, model_a_name, model_b_name)
    print(mcnemar_p)

if __name__ == '__main__':
    main()