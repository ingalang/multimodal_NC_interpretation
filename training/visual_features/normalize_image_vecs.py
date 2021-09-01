from sklearn import preprocessing
import numpy as np
import json

def main():
    with open('../../data/resources/word_to_img_vec_ResNet152V2_10.json', 'r') as file:
        image_model = json.load(file)

    words = []
    vectors = []

    for word, vector in image_model.items():
        words.append(word)
        vectors.append(vector)
    vectors = np.array(vectors)
    print(vectors.shape)

    normalized_vectors = preprocessing.normalize(vectors)
    print(normalized_vectors.shape)

    normalized_vectors_dict = {}
    for word, vector in zip(words, normalized_vectors):
        normalized_vectors_dict[word] = vector.tolist()

    with open('../../data/resources/word_to_img_vec_ResNet152V2_10_norm.json', 'w') as outfile:
        json.dump(normalized_vectors_dict, outfile)

if __name__ == '__main__':
    main()