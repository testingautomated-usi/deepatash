
from pathlib import Path
import pickle
from random import shuffle
import tensorflow as tf
from tensorflow import keras
import os
import json
import numpy as np
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import INPUT_MAXLEN

from config import MODEL

DATASET_DIR = "data"
train_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='train')
train_data, y_train = train_ds['text'], train_ds['label']
test_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='test')
test_data, y_test = test_ds['text'], test_ds['label']

with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

x_train = tokenizer.texts_to_sequences(train_data)
x_test = tokenizer.texts_to_sequences(test_data)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=INPUT_MAXLEN)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=INPUT_MAXLEN)

y_test = keras.utils.to_categorical(y_test, 2)
y_train = keras.utils.to_categorical(y_train, 2)

def retrain(target_x_train, target_y_train, target_x_test, target_y_test):

    print("\nx_train shape:", target_x_train.shape)
    print(target_x_train.shape[0], "train samples")
    print(target_x_test.shape[0], "test samples")
    
    # Load the pre-trained model.
    model = tf.keras.models.load_model(MODEL)
    epochs = 5
    batch_size = 32

    score = model.evaluate(x_test, y_test, verbose=0)
    test_accuracy_before = score[1]

    score = model.evaluate(target_x_test, target_y_test, verbose=0)
    test_accuracy_target_before = score[1]

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(target_x_train, target_y_train, batch_size=batch_size, epochs=epochs)

    score = model.evaluate(x_test, y_test, verbose=0)
    test_accuracy_after =  score[1]

    score = model.evaluate(target_x_test, target_y_test, verbose=0)
    test_accuracy_target_after = score[1]

    return test_accuracy_before, test_accuracy_target_before, test_accuracy_target_after, test_accuracy_after
    


if __name__ == "__main__":

    dst = "../experiments/data/imdb/retrain"
    Path(dst).mkdir(parents=True, exist_ok=True)
    feature_combinations = {"NegCount-PosCount", "NegCount-VerbCount", "PosCount-VerbCount"}


    for features in feature_combinations:
        accs = []
        for i in range(1, 11):
            dst1 = f"../experiments/data/imdb/DeepAtash/target_cell_in_dark/{features}/{i}-nsga2_-features_{features}-diversity_LATENT"
            dst2 = f"../experiments/data/imdb/DeepAtash/target_cell_in_grey/{features}/{i}-nsga2_-features_{features}-diversity_LATENT"
            dst3 = f"../experiments/data/imdb/DeepAtash/target_cell_in_white/{features}/{i}-nsga2_-features_{features}-diversity_LATENT"
            
            inputs = []
            for subdir, _, files in os.walk(dst1, followlinks=False):
                # Consider only the files that match the pattern
                    for json_path in [os.path.join(subdir, f) for f in files if f.startswith("mbr") and f.endswith(".json")]:              
                            with open(json_path) as jf:
                                json_data = json.load(jf)
                            seq = tokenizer.texts_to_sequences([json_data["text"]])
                            padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
                            if json_data["misbehaviour"] == True:
                                inputs.append([padded_texts[0], json_data['expected_label']])
            for subdir, _, files in os.walk(dst2, followlinks=False):
                # Consider only the files that match the pattern
                    for json_path in [os.path.join(subdir, f) for f in files if f.startswith("mbr") and f.endswith(".json")]:              
                            with open(json_path) as jf:
                                json_data = json.load(jf)
                            seq = tokenizer.texts_to_sequences([json_data["text"]])
                            padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
                            if json_data["misbehaviour"] == True:
                                inputs.append([padded_texts[0], json_data['expected_label']])
            for subdir, _, files in os.walk(dst3, followlinks=False):
                # Consider only the files that match the pattern
                    for json_path in [os.path.join(subdir, f) for f in files if f.startswith("mbr") and f.endswith(".json")]:              
                            with open(json_path) as jf:
                                json_data = json.load(jf)
                            seq = tokenizer.texts_to_sequences([json_data["text"]])
                            padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
                            if json_data["misbehaviour"] == True:
                                inputs.append([padded_texts[0], json_data['expected_label']])
            
            shuffle(inputs)
            target_x_train = []
            target_y_train = []
            target_x_test = []
            target_y_test = []

            for idx in range(int(len(inputs)/2)):
                target_x_train.append(inputs[idx][0])
                target_y_train.append(inputs[idx][1])

            for idx in range(int(len(inputs)/2), len(inputs)):
                target_x_test.append(inputs[idx][0])
                target_y_test.append(inputs[idx][1])        

            np.save(f"{dst}/target_x_train_{features}_{i}.npy", target_x_train)
            np.save(f"{dst}/target_x_test_{features}_{i}.npy", target_x_test)

            # convert class vectors to binary class matrices
            target_y_test = keras.utils.to_categorical(target_y_test, 2)
            target_y_train = keras.utils.to_categorical(target_y_train, 2)

            target_y_train = np.concatenate((target_y_train, y_train), axis=0)
            target_x_train =  np.concatenate((target_x_train, x_train), axis=0)
            


            for rep in range(1, 11):
                t0, t1, t2, t3 = retrain(np.array(target_x_train), np.array(target_y_train), np.array(target_x_test), np.array(target_y_test))
                dict_report = {
                    "approach": "After",
                    "features": features,
                    "accuracy test set": t3,
                    "accuracy target test set": t2
                }
                filedst = f"{dst}/report-{features}-after-{i}-{rep}.json"
                with open(filedst, 'w') as f:
                    (json.dump(dict_report, f, sort_keys=True, indent=4))

                dict_report = {
                    "approach": "Before",
                    "features": features,
                    "accuracy test set": t0,
                    "accuracy target test set": t1
                }
                filedst = f"{dst}/report-{features}-before-{i}-{rep}.json"
                with open(filedst, 'w') as f:
                    (json.dump(dict_report, f, sort_keys=True, indent=4))