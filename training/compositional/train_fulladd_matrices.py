from models.full_add_composition import get_embeddings, build_semantic_space
from models.utils import load_data, remove_non_existing
from dissect.src.composes.composition.full_additive import FullAdditive

def train_fulladd_matrices(train_data, model, second_model=None):
    train_data_exists_in_model = remove_non_existing(train_data, model, filter_on=['mod', 'head', 'compound'])
    print(f'len of train data after one filtering inside train_fulladd_matrices: {len(train_data_exists_in_model)}')

    if not isinstance(second_model, type(None)):
        train_data_exists_in_model = remove_non_existing(train_data_exists_in_model, second_model, filter_on=['mod', 'head', 'compound'])
    print(f'len of train data after two filterings inside train_fulladd_matrices: {len(train_data_exists_in_model)}')


    new_train = train_data_exists_in_model.drop_duplicates(subset=['nc_mod'])
    new_train = new_train.drop_duplicates(subset=['nc_head'])
    new_train = new_train.drop_duplicates(subset=['compound'])

    space_train_mod_embeddings, _ = get_embeddings(new_train['nc_mod'], model)
    space_train_head_embeddings, _ = get_embeddings(new_train['nc_head'], model)
    space_train_compound_embeddings, _ = get_embeddings(new_train['compound'], model)

    assert (len(space_train_mod_embeddings) == len(space_train_head_embeddings) == len(
        space_train_compound_embeddings)), 'size of spaces must be the same'
    train_support = len(space_train_mod_embeddings)

    mod_space = build_semantic_space(space_train_mod_embeddings, list(new_train['nc_mod']))
    head_space = build_semantic_space(space_train_head_embeddings, list(new_train['nc_head']))
    compound_space = build_semantic_space(space_train_compound_embeddings, list(new_train['compound']))
    print(type(mod_space), type(compound_space))

    comp_train_data = [(mod, head, compound)
                       for mod, head, compound
                       in zip(new_train['nc_mod'], new_train['nc_head'], new_train['compound'])]

    comp_model = FullAdditive()
    comp_model._regression_learner._intercept = False
    comp_model.train(comp_train_data, (mod_space, head_space), compound_space)
    return comp_model._mat_a_t.get_mat(), comp_model._mat_b_t.get_mat(), train_support