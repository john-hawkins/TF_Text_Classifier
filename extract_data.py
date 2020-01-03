#
# EXTRACT THE IMDB Review Sentiment Data
#

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd

dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
 
# THIS IS MEANT TO BE AN INSTANCE OF tfds.features.text.SubwordTextEncoder
# IN MY VERSION IT WAS tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder
# NEEDED BECAUSE THIS DATASET IS STORED IN AN ENCODED FORMAT
encoder = info.features['text'].encoder


# WRITE THESE TWO DATASETS OUT AS CSV FILES

train_df = pd.DataFrame(columns=["review", "target"])

for element in train_dataset:
    train_df = train_df.append({
        "review": encoder.decode(element[0].numpy().tolist()),
        "target": element[1].numpy()
    }, ignore_index=True)


test_df = pd.DataFrame(columns=["review", "target"])

for element in test_dataset:
    test_df = test_df.append({
        "review": encoder.decode(element[0].numpy().tolist()),
        "target": element[1].numpy()
    }, ignore_index=True)


# WRITE TO DISK
train_df.to_csv("data/train_df.csv", index=False, header=True)
test_df.to_csv("data/test_df.csv", index=False, header=True)



