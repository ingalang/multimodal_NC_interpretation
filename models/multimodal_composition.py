import numpy as np
from models.full_add_composition import compose_vector, get_embedding
from models.utils import remove_non_existing, combine_vectors
from training.compositional.train_fulladd_matrices import train_fulladd_matrices

def get_multimodal_fulladd_vectors(train_data,
                                   test_data,
                                   val_data,
                                   text_model,
                                   image_model,
                                   order_of_composition,
                                   third_model=None):
    possible_orders = ['similar_first', 'opposite_first']
    assert order_of_composition in possible_orders, \
        f'Parameter order_of_composition must be one of these: {possible_orders}'

    final_train_vectors, final_test_vectors, final_val_vectors = [], [], []

    if not isinstance(third_model, type(None)):
        train_data_filtered = remove_non_existing(train_data, third_model, filter_on=['mod', 'head', 'compound'])
        print(f'filtered')
    else:
        train_data_filtered = train_data

    train_data_filtered = remove_non_existing(train_data_filtered, text_model, filter_on=['mod', 'head', 'compound'])
    print(f'length of train_data_filtered after filtering on text model: {len(train_data_filtered)}')
    train_data_filtered = remove_non_existing(train_data_filtered, image_model, filter_on=['mod', 'head', 'compound'])
    print(f'length of train_data_filtered after filtering on image model: {len(train_data_filtered)}')

    if order_of_composition == 'similar_first':
        # if the order of composition is `similar first`, we first combine vectors from the same modalities
        # (meaning mods + heads in the text modality go together, and mods + heads in the image modality go together
        text_mat_a, text_mat_b, text_train_support = train_fulladd_matrices(
            train_data=train_data_filtered,
            model=text_model,
            second_model=image_model)
        image_mat_a, image_mat_b, image_train_support = train_fulladd_matrices(
            train_data=train_data_filtered,
            model=image_model,
            second_model=text_model)
        print(f'TEXT TRAIN SUPPORT: {text_train_support}')
        print(f'IMAGE TRAIN SUPPORT: {image_train_support}')

        train_samples_processed = 0

        for mod, head in zip(train_data['nc_mod'], train_data['nc_head']):
            train_samples_processed += 1
            text_mod_vector = text_model[mod] if mod in text_model else np.zeros(300)
            text_head_vector = text_model[head] if head in text_model else np.zeros(300)
            composed_text_vector = compose_vector(text_mat_a, text_mat_b, text_mod_vector, text_head_vector)

            if mod in image_model and head in image_model:
                image_mod_vector, image_head_vector = image_model[mod], image_model[head]
                composed_img_vector = compose_vector(image_mat_a, image_mat_b, image_mod_vector, image_head_vector)
                multimodal_vector = combine_vectors(composed_text_vector, composed_img_vector, 'add')
                final_train_vectors.append(multimodal_vector)
            else:
                print('appending text vector only')
                final_train_vectors.append(composed_text_vector)
        print(f'processed {train_samples_processed} train samples in get_multimodal_fulladd_vectors')

        for mod, head in zip(test_data['nc_mod'], test_data['nc_head']):
            text_mod_vector = text_model[mod] if mod in text_model else np.zeros(300)
            text_head_vector = text_model[head] if head in text_model else np.zeros(300)
            composed_text_vector = compose_vector(text_mat_a, text_mat_b, text_mod_vector, text_head_vector)

            if mod in image_model and head in image_model:
                image_mod_vector, image_head_vector = image_model[mod], image_model[head]
                composed_img_vector = compose_vector(image_mat_a, image_mat_b, image_mod_vector, image_head_vector)
                multimodal_vector = combine_vectors(composed_text_vector, composed_img_vector, 'add')
                final_test_vectors.append(multimodal_vector)
            else:
                final_test_vectors.append(composed_text_vector)

        for mod, head in zip(val_data['nc_mod'], val_data['nc_head']):
            text_mod_vector = text_model[mod] if mod in text_model else np.zeros(300)
            text_head_vector = text_model[head] if head in text_model else np.zeros(300)
            composed_text_vector = compose_vector(text_mat_a, text_mat_b, text_mod_vector, text_head_vector)

            if mod in image_model and head in image_model:
                image_mod_vector, image_head_vector = image_model[mod], image_model[head]
                composed_img_vector = compose_vector(image_mat_a, image_mat_b, image_mod_vector, image_head_vector)
                multimodal_vector = combine_vectors(composed_text_vector, composed_img_vector, 'add')
                final_val_vectors.append(multimodal_vector)
            else:
                final_val_vectors.append(composed_text_vector)

    elif order_of_composition == 'opposite_first':

        multimodal_train_vecs = {}

        # getting multimodal mod, head, and comound vectors
        for mod, head, compound in zip(train_data_filtered['nc_mod'], train_data_filtered['nc_head'], train_data_filtered['compound']):
            if mod not in multimodal_train_vecs:
                text_vector = get_embedding(mod, text_model)
                img_vector = get_embedding(mod, image_model)
                multimodal_vector = combine_vectors(text_vector, img_vector, 'add')
                multimodal_train_vecs[mod] = multimodal_vector
            if head not in multimodal_train_vecs:
                text_vector = get_embedding(head, text_model)
                img_vector = get_embedding(head, image_model)
                multimodal_vector = combine_vectors(text_vector, img_vector, 'add')
                multimodal_train_vecs[mod] = multimodal_vector
            if compound not in multimodal_train_vecs:
                text_vector = get_embedding(compound, text_model)
                img_vector = get_embedding(compound, image_model)
                multimodal_vector = combine_vectors(text_vector, img_vector, 'add')
                multimodal_train_vecs[compound] = multimodal_vector

        # training fulladd matrices with multimodal vectors
        multimodal_mat_a, multimodal_mat_b, multimodal_train_support = train_fulladd_matrices(
            train_data=train_data,
            model=multimodal_train_vecs)
        print(f'MULTIMODAL TRAIN SUPPORT: {multimodal_train_support}')

        # getting the final train, test, and val feature vectors using the matrices trained above
        for mod, head in zip(train_data['nc_mod'], train_data['nc_head']):
            mod_vector = combine_vectors(text_model[mod], image_model[mod], 'add')
            head_vector = combine_vectors(text_model[head], image_model[head], 'add')
            composed_multimodal_vector = compose_vector(multimodal_mat_a, multimodal_mat_b, mod_vector, head_vector)
            final_train_vectors.append(composed_multimodal_vector)

        for mod, head in zip(test_data['nc_mod'], test_data['nc_head']):
            mod_vector = combine_vectors(text_model[mod], image_model[mod], 'add')
            head_vector = combine_vectors(text_model[head], image_model[head], 'add')
            composed_multimodal_vector = compose_vector(multimodal_mat_a, multimodal_mat_b, mod_vector, head_vector)
            final_test_vectors.append(composed_multimodal_vector)

        for mod, head in zip(val_data['nc_mod'], val_data['nc_head']):
            mod_vector = combine_vectors(text_model[mod], image_model[mod], 'add')
            head_vector = combine_vectors(text_model[head], image_model[head], 'add')
            composed_multimodal_vector = compose_vector(multimodal_mat_a, multimodal_mat_b, mod_vector, head_vector)
            final_val_vectors.append(composed_multimodal_vector)

    return final_train_vectors, final_test_vectors, final_val_vectors




