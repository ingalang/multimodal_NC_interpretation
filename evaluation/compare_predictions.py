import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds_a', type=str, required=True, help='A file to evaluate')
    parser.add_argument('--preds_b', type=str, required=True, help='Another file to compare with the first file')

    args = parser.parse_args()
    filename_a, filename_b = args.preds_a, args.preds_b

    assert(filename_a.endswith('.csv')), f'{filename_a} is not a csv file! Only csv files are accepted.'
    assert(filename_b.endswith('.csv')), f'{filename_b} is not a csv file! Only csv files are accepted.'

    data_a = pd.read_csv(filename_a)
    data_b = pd.read_csv(filename_b)

    print(data_b.predicted_id)

    metadata_a = get_file_metadata(filename_a)
    metadata_b = get_file_metadata(filename_b)

    print_single_confusion_matrix(data_a)

    print(metadata_a)
    print(metadata_b)

if __name__ == '__main__':
    main()