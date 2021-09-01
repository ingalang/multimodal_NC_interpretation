import argparse
import io
import os
import json
import itertools
import PIL.Image
import random
import requests
import shutil

import numpy as np

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19_input

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnetv2_input

from tensorflow.keras.preprocessing import image


def get_img_from_url(url: str):
    try:
        response = requests.get(url, timeout=3)
        image_bytes = io.BytesIO(response.content)
        img = PIL.Image.open(image_bytes)
        if img.mode != 'CMYK':
            return img
        else:
            return None
    except:
        return None

def get_representation_from_url(url: str, model, model_name):
    img = get_img_from_url(url)
    img = img.resize((224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)

    if model_name == 'VGG19':
        img_data = preprocess_vgg19_input(img_data)
    elif model_name == 'ResNet152V2':
        img_data = preprocess_resnetv2_input(img_data)
    else: raise ValueError(f'Unknown vision model name: {model_name}')

    feature_vector = model.predict(img_data)
    return feature_vector.flatten()

def get_avg_representation_for_word(word, word_to_urls, model, model_name, num_images, save_example_images=3):
    print(f'Getting average representation for {word}')
    urls = word_to_urls[word]
    if len(urls) < num_images:
        return None
    print(f'Got {len(urls)} urls for this word: {word}')
    images = []

    for url in urls:
        img = get_img_from_url(url)
        if img is not None:
            images.append(img)
        if len(images) >= num_images:
            break
    if len(images) < num_images:
        print(f'Could not find enough images for this word: {word}')
        return None

    print(f'Found {len(images)} images for this word: {word}')

    if save_example_images:
        example_images = random.sample(images, save_example_images)
        title = '_'.join(word.split())
        for i, img in enumerate(example_images):
            print(img.mode)
            if img.mode != 'CMYK':
                if os.path.exists(f'example_images/{title}'):
                    img.save(f'example_images/{title}/{i}.png')
                else:
                    os.mkdir(f'example_images/{title}')
                    img.save(f'example_images/{title}/{i}.png')

    representations = []
    steps = 1
    title = '_'.join(word.split())
    img_dir = f'{title}'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    for img in images:
        print(f'Processing image number {steps}')
        try:
            img = img.resize((224, 224))

            ##################################################################
            img_path = f'{title}/{steps}.png'
            # TODO bruk dette senere hvis du trenger å lagre dem og så slette
            img.save(img_path)
            ##################################################################

            img = image.load_img(img_path, color_mode='rgb', target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)

            if model_name == 'VGG19':
                img_data = preprocess_vgg19_input(img_data)
            elif model_name == 'ResNet152V2':
                img_data = preprocess_resnetv2_input(img_data)
            else:
                raise ValueError(f'Unknown vision model name: {model_name}')

            vgg16_feature = model.predict(img_data)
            representations.append(vgg16_feature.flatten())

        except:
            if os.path.exists(img_dir):
                shutil.rmtree(img_dir)
            return None
        steps += 1
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    representations = np.array(representations)
    print(representations.shape)
    return np.mean(representations, axis=0)

def get_word_to_urls_dict(word_to_synsets, synset_to_urls):
    word_to_urls = {}

    for word, synsets in word_to_synsets.items():
        urls = []
        for s in synsets:
            urls.append(synset_to_urls[s])
        urls = list(itertools.chain(*urls))
        word_to_urls[word] = urls
    return word_to_urls

def process_batch(word_list, word_to_urls, model, model_name, num_images, save_examples):
    word_to_img_vec = {}
    images_not_found = []
    for word in word_list:
        vector = get_avg_representation_for_word(word=word,
                                                 word_to_urls=word_to_urls,
                                                 model=model,
                                                 model_name=model_name,
                                                 num_images=num_images,
                                                 save_example_images=save_examples)
        if vector is not None:
            word_to_img_vec[word] = vector.tolist()
        else:
            images_not_found.append(word)
    return word_to_img_vec, images_not_found

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=False, default='ResNet152V2',
                        help='Which model to use. Options: VGG19, ResNet152V2. Default=ResNet152V2')
    parser.add_argument('--word_to_urls', type=str, required=False, default= '../../data/resources/word_to_urls.json',
                        help='Path to .json file of the format {\'word\' : [list of image urls]. '
                             'If not provided, it will need be created from a words_to_synsets and a synset_to_urls file.')
    parser.add_argument('--word_to_synsets', type=str, required=False, default='../../data/resources/word_to_synsets.json',
                        help='Path to .json file of the format {\'word\' : [list of synsets]')
    parser.add_argument('--synset_to_urls', type=str, required=False, default='../../data/resources/synset_to_urls.json',
                        help='Path to .json file of the format {\'synset\' : [list of image urls]')
    parser.add_argument('--batch_size', type=int, required=False, default=5,
                        help='Number of words to get representations for '
                             'between each time the representations file is saved. Default: 10')
    parser.add_argument('--num_images', type=int, required=False, default=1,
                        help='Number of images to average for each representation. '
                             'Will discard all words that do not have enough available images to reach this number. Default: 1')
    parser.add_argument('--save_examples', type=int, required=False, default=0,
                        help='Number of example images to save per word. Default: 0.')

    args = parser.parse_args()
    model_name, word_to_urls_name, word_to_synsets_name, synset_to_urls_name, batch_size, num_images, save_examples = \
        args.model, args.word_to_urls, args.word_to_synsets, args.synset_to_urls, args.batch_size, args.num_images, args.save_examples

    try:
        with open(word_to_urls_name, 'r') as infile:
            word_to_urls = json.load(infile)
    except:
        with open(word_to_synsets_name, 'r') as infile:
            word_to_synsets = json.load(infile)
        with open(synset_to_urls_name, 'r') as infile:
            synset_to_urls = json.load(infile)

        word_to_urls = get_word_to_urls_dict(word_to_synsets, synset_to_urls)

        with open(word_to_urls_name, 'w') as outfile:
            json.dump(word_to_urls, outfile)

    if model_name == 'ResNet152V2':
        model = ResNet152V2(weights='imagenet', include_top=False)

    elif model_name == 'VGG19':
        model = VGG19(weights='imagenet', include_top=False)
    else:
        raise ValueError(f'Unknown vision model name: {model_name}')

    word2imgvec_filename = f'word_to_img_vec_{model_name}_{num_images}.json'

    try:
        with open(f'../../data/resources/images_not_found_{num_images}.json', 'r') as file:
            images_not_found = json.load(file)
    except:
        images_not_found= []

    word_to_vec_filepath = f'../../data/resources/{word2imgvec_filename}'
    if os.path.exists(word_to_vec_filepath):
        with open(word_to_vec_filepath, 'r') as file:
            word_to_img_vec = json.load(file)
            print(f'Opened {word2imgvec_filename} file')
    else:
        word_to_img_vec = {}
        print(f'Could not find {word2imgvec_filename} file, made a new one')

    word_list = [w for w in word_to_urls.keys() if w not in word_to_img_vec and w not in images_not_found]

    while word_list:
        words_to_process = [w for w in word_list[:batch_size] if w not in images_not_found]
        new_word_to_vec, batch_imgs_not_found = process_batch(word_list=words_to_process,
                                                              word_to_urls=word_to_urls,
                                                              model=model,
                                                              model_name=model_name,
                                                              num_images=num_images,
                                                              save_examples=save_examples)
        images_not_found += batch_imgs_not_found
        for key, value in new_word_to_vec.items():
            print(key, value[:3])

        word_to_img_vec.update(new_word_to_vec)
        del new_word_to_vec
        del word_list[:batch_size]
        with open(f'../../data/resources/{word2imgvec_filename}', 'w') as outfile:
            json.dump(word_to_img_vec, outfile)
            print('Saved the thing')
        with open(f'../../data/resources/images_not_found_{num_images}.json', 'w') as file:
            json.dump(images_not_found, file)

if __name__ == '__main__':
    main()
