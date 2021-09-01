import pandas as pd
pd.options.mode.chained_assignment = None
import json
from models.full_add_composition import get_embedding
from models.utils import get_model
import argparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_wordlist(model_1, model_2=None):
    with open('../data/resources/word_to_img_vec_ResNet152V2_100_pca_300.json', 'r') as file:
        corpus = json.load(file)
    wordlist = []
    for w in corpus.keys():
        try:
            get_embedding(w, model_1)
            if model_2 is not None:
                get_embedding(w, model_2)
            wordlist.append(w)
        except:
            pass
    return wordlist

def tsne_plot(model, wordlist, plot_title=None, save_file_as=None):
    "Creates a TSNE model and plots it"
    #plt.interactive(False)

    labels = []
    tokens = []

    for word in wordlist:
        try:
            tokens.append(get_embedding(word, model))
            labels.append(word)
        except:
            pass

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    if plot_title is not None:
        plt.title(plot_title)

    # plot will only scatter every n tokens from the list, so the plot is readable.
    print(f'Scattering {len(labels)} words')
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom',
                        fontsize=8)
    if save_file_as is not None:
        plt.savefig(save_file_as, dpi=300)

    plt.show()

def get_nearest_neighbors(target, wordlist, model, n):
    neighbors = {}
    target_embedding = get_embedding(target, model)

    for w in wordlist:
        try:
            embedding = get_embedding(w, model)
            vector_matrix = np.vstack((target_embedding, embedding))
            similarity = cosine_similarity(vector_matrix)[0][1]
            neighbors[w] = similarity
        except:
            pass
    del neighbors[target]
    sorted_by_similarity = sorted(neighbors.items(), key=lambda item: item[1], reverse=True)
    return sorted_by_similarity[:n]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot_embeddings', required=False, type=str, default='True')
    parser.add_argument('--model_1_name', required=False, type=str, default='ResNet_10_norm')
    parser.add_argument('--model_2_name', required=False, type=str, default='ResNet_10')
    parser.add_argument('--print_nearest_neighbors', required=False, type=bool, default=True)
    parser.add_argument('--n_neighbors', required=False, type=int, default=5)

    args = parser.parse_args()

    plot_embeddings, model_1_name, model_2_name, print_n_neighbors, n = \
        args.plot_embeddings, args.model_1_name, args.model_2_name, args.print_nearest_neighbors, args.n_neighbors

    print('plot embeddings', plot_embeddings)
    model_1 = get_model(model_1_name)
    model_2 = get_model(model_2_name)
    model_3 = get_model('word2vec')

    words = get_wordlist(model_1,model_2)

    words_to_plot = random.sample(words, 200)

    if plot_embeddings == 'True':
        tsne_plot(model_1, words_to_plot, plot_title='ResNet152V2, 10-avg, PCA-300, normalized to [-1, 1]', save_file_as='RN_10_300_norm_plot.png')
        tsne_plot(model_2, words_to_plot, plot_title='ResNet152V2, 10-avg, PCA-300, not normalized', save_file_as='RN_10_300_plot_new.png')
        tsne_plot(model_3, words_to_plot, plot_title='word2vec, GoogleNews300-neg', save_file_as='word2vec_plot_new.png')

    if print_n_neighbors:
        test_words = ['desktop', 'brand', 'surface', 'stone',
                      'reservoir', 'cadet', 'chenille', 'pizza', 'cocktail', 'unionist']
        #test_words = random.sample(words, 10)

        for word in test_words:
            print(f'### Getting nearest neighbors for {word}\n')
            print(f'Model 1: {model_1_name}')
            nearest_neighbors_1 = get_nearest_neighbors(word, words, model_1, n)
            for (k, v) in nearest_neighbors_1:
                print(k, ': ', v)
            print()
            print(f'Model 2: {model_2_name}')
            nearest_neighbors_2 = get_nearest_neighbors(word, words, model_2, n)
            for (k, v) in nearest_neighbors_2:
                print(k, ': ', v)
            print()

if __name__ == '__main__':
    main()