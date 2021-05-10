import json, copy
from collections import defaultdict
import re
import math

def _avg_sent_per_compound(sentence_dict: dict):
    num_sentences = 0
    for k, v in sentence_dict.items():
        assert isinstance(v, list), 'not all values in the dictionary are lists'
        num_sentences += len(v)
    return num_sentences/len(sentence_dict)

def _avg_len_per_sentence(sentence_dict: dict):
    num_sentences = 0
    combined_sent_length = 0
    for k, v in sentence_dict.items():
        assert isinstance(v, list), 'not all values in the dictionary are lists'
        num_sentences += len(v)
        total_words = [w for sent in v for w in sent.split()]
        combined_sent_length += len(total_words)
    return combined_sent_length/num_sentences

def _number_of_empty_sentence_lists(sentence_dict: dict) -> int:
    num_of_empty_lists = 0
    for v in sentence_dict.values():
        if len(v) == 0:
            num_of_empty_lists += 1
    return num_of_empty_lists

def print_sentence_stats(sentence_dict: dict, msg='sentence stats'):
    print('{0:#^50}'.format(' ' + msg + ' '))
    print(f'Dictionary contains sentences for {len(sentence_dict)} compounds.')
    print(f'Each compound has an average of {_avg_sent_per_compound(sentence_dict)} sentences.')
    print(f'Sentence length micro average is {_avg_len_per_sentence(sentence_dict)} words per sentence.')
    print(f'Number of compounds with no sentences: {_number_of_empty_sentence_lists(sentence_dict)}.')

def clean_sentences(sentence_dict: dict) -> dict:
    url_reg = re.compile(r'http\S+|\[\w*\]|[^\w\s!?:.,\-/]')
    space_reg = re.compile(r'\s+')
    cleaned_sentences = defaultdict(list)
    for compound, sentences in sentence_dict.items():
        for s in sentences:
            new_s = re.sub(url_reg, '', s)
            new_s = re.sub(space_reg, '', new_s)
            cleaned_sentences[compound].append(new_s)
    return cleaned_sentences


def filter_sentences(sentence_dict: dict, desired_length: int, num_sentences: int) -> dict:
    """
    :param sentence_dict: dictionary containing compound : [list of sentences]
    :param desired_length: desired length of sentences,
                           will choose sentences closest to this length over longer/shorter ones
    :param num_sentences: number of sentences to keep per compound
    :return: dict of filtered sentences
    """
    filtered_sentences = defaultdict(list)
    for compound, sentences in sentence_dict.items():
        if not sentences:
            print(f'No sentences listed for "{compound}".')
            continue
        regex = f'{compound}'
        potential_sentences = []
        for s in sentences:
            if re.search(regex, s, re.IGNORECASE) is not None:
                potential_sentences.append(s)
        potential_sentences = sorted(potential_sentences, key=lambda x: math.fabs(len(x.split())-desired_length))
        filtered_sentences[compound].append(potential_sentences[:num_sentences])

    return sentence_dict


def main():
    # TODO bytte hvordan du åpner fila, må gjøres før du publiserer
    try:
        with open('/Users/ingalang/Documents/Malta/Semester2/Thesis/compound_sents_10mai_1233.json', 'r') as infile:
            harvested_sentences = json.load(infile)
    except:
        print('Cannot open file')

    print_sentence_stats(harvested_sentences, msg='sentence stats before filtering')
    filter_sentences(sentence_dict=harvested_sentences, desired_length=22, num_sentences=1)
    print(len(harvested_sentences['creditor meeting']))
    print(harvested_sentences['creditor meeting'])

if __name__ == '__main__':
    main()