from builtins import object  # pylint: disable=redefined-builtin

import tensorflow_datasets as tfds
import tensorflow as tf
import itertools
import numpy as np

class CustomInferenceModel(object):
    """
    This is a template for Python inference model scoring code.
    It loads the custom model, performs any necessary preprocessing or feature engineering,
    and then performs predictions.

    Note: If your model is a binary classification model, you will likely want your predict
           function to use `predict_proba`, whereas for regression you will want to use `predict`
    """

    def __init__(self, path_to_model="custom_model.h5"):
        """Load the Tensorflow Model"""
 
        self.model = tf.keras.models.load_model(path_to_model)
        self.encoder = tfds.features.text.SubwordTextEncoder.load_from_file('encoder.dat')

    def encode_review(self, x):
        """ Helper function for pre-processing """
        x['review'] = self.encoder.encode(x['review'])
        return x

    def pad_review(self, x):
        """ Helper function for pre-processing """
        newx = x.copy()
        result = np.zeros(self.array_length, dtype=np.int)
        thisX = x['review']
        result[0:len(thisX)] = thisX
        newx['review'] = result
        return newx

    def preprocess_features(self, X):
        """ For this model we need to encode the 'review' column using the embeded text encoder. Then extract the resulting column into a numpy array."""
        temp = X.apply(self.encode_review, axis=1)
        self.num_records = len(X)
        self.array_length = temp['review'].map(len).max()
        temp2 =  temp.apply(self.pad_review, axis=1)
        flattened = list(itertools.chain.from_iterable(temp2['review'].values))
        npflat = np.array(flattened)
        rez = npflat.reshape(self.num_records,self.array_length)
        return rez 

    def _determine_positive_class_index(self, positive_class_label):
        """Find index of positive class label to interpret predict_proba output"""
        return 0

    def predict(self, X, positive_class_label=None, negative_class_label=None, **kwargs):
        """
        Predict with the custom model.
        """
        X = self.preprocess_features(X)
        predictions = self.model.predict(X)
        predictions = [
                {
                    positive_class_label: str(prediction[0]),
                    negative_class_label: str(1 - prediction[0]),
                }
                for prediction in predictions
        ]
        return predictions


