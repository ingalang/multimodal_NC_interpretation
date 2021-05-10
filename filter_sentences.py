import json

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

def print_sentence_stats(sentence_dict: dict, msg='sentence stats'):
    print('{0:#^50}'.format(' ' + msg + ' '))
    print(f'Dictionary contains sentences for {len(sentence_dict)} compounds.')
    print(f'Each compound has an average of {_avg_sent_per_compound(sentence_dict)} sentences.')
    print(f'Sentence length micro average is {_avg_len_per_sentence(sentence_dict)} words per sentence.')

def main():

    try:
        with open('/Users/ingalang/Documents/Malta/Semester2/Thesis/compound_sents_10mai_1233.json', 'r') as infile:
            harvested_sentences = json.load(infile)
    except:
        print('Cannot open file')

    print_sentence_stats(harvested_sentences, msg='sentence stats before filtering')

if __name__ == '__main__':
    main()