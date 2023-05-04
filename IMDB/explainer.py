from alibi.explainers import IntegratedGradients
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import logging as log

import pickle
from config import MODEL
from config import VOCAB_SIZE, INPUT_MAXLEN


model = tf.keras.models.load_model(MODEL)

layer = model.layers[1]

n_steps = 50
method = "gausslegendre"
ig  = IntegratedGradients(model,
                          layer=layer,
                          n_steps=n_steps,
                          method=method)


# loading
with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def explain_integrated_gradiant(X):
    #Predictions vector
    seq = tokenizer.texts_to_sequences([X])
    padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
    predictions = model.predict(padded_texts).argmax(axis=1)

    explanation = ig.explain(padded_texts,
                            baselines=None,
                            target=predictions,
                            attribute_to_layer_inputs=False)

    # Get attributions values from the explanation object
    attrs = explanation.attributions[0]
    attrs = attrs.sum(axis=2)
    expl = attrs[0]
    expl[expl<0] = 0

    processed = process_text_contributions(padded_texts[0],expl)  

    return  processed


def process_text_contributions(data, contributions):
    """
    process contributions by transforming the text to vec 
    and assing contributions to each word
    :param data: original texts
    :param contributions: text contributions
    :return: The processed contributions
    """

    processed_contribution = np.zeros(shape=(VOCAB_SIZE), dtype=float)
    for idx2 in range(len(data)):
        word_index = data[idx2]
        processed_contribution[word_index] = contributions[idx2] 
    return processed_contribution   


