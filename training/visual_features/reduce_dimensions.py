import argparse
import json
import numpy as np
from sklearn.decomposition import PCA

def get_wordlist_and_vector_matrix(word_to_img_vec: dict):
    wordlist = []
    vectors = []
    for word, vector in word_to_img_vec.items():
        wordlist.append(word)
        vectors.append(vector)
    vector_matrix = np.array(vectors)
    return wordlist, vector_matrix

#TODO kan hende du ikke kan bruke tsne i det hele tatt????
def reduce_dimensions(word_to_img_vec: dict, technique: str, dimensions: int):
    wordlist, vector_matrix = get_wordlist_and_vector_matrix(word_to_img_vec)

    if technique == 'pca':
        pca = PCA(n_components=dimensions)
        reduced_dim_matrix = pca.fit_transform(vector_matrix)
    else:
        raise NotImplementedError('Only the PCA technique has been implemented')

    final_dict = {}
    print('Shape of reduced_dim_matrix: ', reduced_dim_matrix.shape)
    for word, vector in zip(wordlist, reduced_dim_matrix):
        assert len(vector) == dimensions, (f'Something happened. Dimensions were set to {dimensions} '
                                           f'but vectors are {len(vector)}')
        final_dict[word] = vector.tolist()
    return final_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=False,
                        default='../../data/resources/word_to_img_vec_ResNet152V2_10_norm.json',
                        help='File to read words and vectors from. Must be a .json file. '
                             'Output will be saved with same base name but including an extension '
                             'for new dimensions and reduction technique')
    parser.add_argument('--technique', type=str, required=False, default='pca',
                        help='Technique with which to reduce dimensionality of the vectors. '
                             'Default: pca')
    parser.add_argument('--dims', type=int, required=False, default=300,
                        help='Dimensions to reduce to. Default: 300')

    args = parser.parse_args()
    filename, technique, dims = args.infile, args.technique, args.dims

    assert(filename.endswith('.json')), ('File must be .json file!')

    base_filename = filename.split('.json')[0]

    with open(filename, 'r') as infile:
        word_to_img_vecs = json.load(infile)

    reduced_dims_dict = reduce_dimensions(word_to_img_vec=word_to_img_vecs,
                                          technique=technique,
                                          dimensions=dims)

    new_filename = base_filename + f"_{technique}_{dims}.json"

    with open(new_filename, 'w') as outfile:
        json.dump(reduced_dims_dict, outfile)


if __name__ == '__main__':
    main()