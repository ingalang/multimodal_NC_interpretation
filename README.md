# multimodal_NC_interpretation

This repository contains the code for my master's thesis called "Comparing Linguistic and Visuo-Linguistic Representations for Noun-Noun Compound Relation Classification in English". It includes code I wrote to collect and pre-process data as well as code to run my models and compare the results.
Below, I give an overview of the files contained in the project and how they work.

## data (folder)

### filter_tratz_data.py
Script that filters the Tratz (2011) data, removing the lexicalized and person classes. Requires having the Tratz_2011_data saved under "resources". Saves the data as Tratz_2011_comp_binary.

### get_synsets.py
Script that employs heuristic to select synsets from ImageNet based on words from the Tratz_2011_comp_binary data.

### image_stats.py
Shows descriptive statistics of the images collected from ImageNet.

### collection (folder)
#### sentence_harvester.py
Script for harvesting sentences from the web for the BERT models. 

### filter_sentences.py
Script for filtering and cleaning sentences harvested from the web.

### resources (folder)
### data_stats.py
Script that provides an overview of descriptive statistics of the Tratz (2011) dataset. Requires having the Tratz_2011_comp_binary folder saved under "resources".

## models (folder)
### baseline.py
Baseline classifiers.

### full_add_composition.py
Contains tools for composing vectors with the fullAdd model.

### multimodal_composition.py
Contains tools for composing vectors multimodally (visuo-linguistically).

### utils.py
Contains tools for loading data and models, etc.

### classification.py
Script that runs classifiers and saves predictions to files.

## evaluation (folder)
### compare_svm_predictions.py
Takes two SVM results files and prints an extensive statistics report along with statistical tests and p-values, as well as F-scores per class and weighted F1 and accuracy. Requires having a "results" folder at the same level as "evaluation", containing the two SVM prediction files you want to compare.

### inspect_embeddings.py
Script for plotting per-relation F1 scores and word/image embeddings.

## training (folder)
### compositional (folder
#### train_fulladd_matrices.py
Utils for training fulladd matrices.

### visual_features (folder)
#### get_visual_embeddings.py
Uses collected URLs for the words in the Tratz (2011) data and obtains feature vectors for each image using a ResNet152 model.

#### normalize_image_vecs.py
Script for normalizing the ResNet vectors.

#### reduce_dimensions.py
Script for reducing dimensions of ResNet vectors.




