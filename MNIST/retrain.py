
from pathlib import Path
from random import shuffle
import tensorflow as tf
from tensorflow import keras
import os
import json
import numpy as np


from config import MODEL, NUM_CLASSES

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_test = x_test.astype("float32") / 255
x_train = x_train.astype("float32") / 255

y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)


def retrain(target_x_train, target_y_train, target_x_test, target_y_test):
    
    # Make sure images have shape (28, 28, 1)
    target_x_train = np.expand_dims(target_x_train, -1)
    target_x_test = np.expand_dims(target_x_test, -1)

    print("\nx_train shape:", target_x_train.shape)
    print(target_x_train.shape[0], "train samples")
    print(target_x_test.shape[0], "test samples")

    # Load the pre-trained model.
    model = tf.keras.models.load_model(MODEL)
    epochs = 6
    batch_size = 128

    score = model.evaluate(x_test, y_test, verbose=0)
    test_accuracy_before = score[1]

    score = model.evaluate(target_x_test, target_y_test, verbose=0)
    test_accuracy_target_before = score[1]

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.fit(target_x_train, target_y_train, batch_size=batch_size, epochs=epochs)

    score = model.evaluate(x_test, y_test, verbose=0)
    test_accuracy_after =  score[1]

    score = model.evaluate(target_x_test, target_y_test, verbose=0)
    test_accuracy_target_after = score[1]

    return test_accuracy_before, test_accuracy_target_before, test_accuracy_target_after, test_accuracy_after
    


if __name__ == "__main__":

    dst = "../experiments/data/mnist/retrain"
    Path(dst).mkdir(parents=True, exist_ok=True)
    feature_combinations = {"Moves-Bitmaps", "Moves-Orientation", "Orientation-Bitmaps"}

    for features in feature_combinations:
        for i in range(1, 11):
            dst1 = f"../experiments/data/mnist/DeepAtash/target_cell_in_dark/{features}/{i}-nsga2_-features_{features}-diversity_LATENT"
            dst2 = f"../experiments/data/mnist/DeepAtash/target_cell_in_grey/{features}/{i}-nsga2_-features_{features}-diversity_LATENT"
            dst3 = f"../experiments/data/mnist/DeepAtash/target_cell_in_white/{features}/{i}-nsga2_-features_{features}-diversity_LATENT"
            
            inputs = []
            for subdir, _, files in os.walk(dst1, followlinks=False):
                # Consider only the files that match the pattern
                    for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:    
                            json_path = svg_path.replace(".svg", ".json")            
                            with open(json_path) as jf:
                                json_data = json.load(jf)

                            npy_path = svg_path.replace(".svg", ".npy") 
                            image = np.load(npy_path)
                            if json_data["misbehaviour"] == True:
                                inputs.append([np.squeeze(image), json_data['expected_label']])
            for subdir, _, files in os.walk(dst2, followlinks=False):
                # Consider only the files that match the pattern
                    for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:  
                            json_path = svg_path.replace(".svg", ".json")            
                            with open(json_path) as jf:
                                json_data = json.load(jf)

                            npy_path = svg_path.replace(".svg", ".npy") 
                            image = np.load(npy_path)
                            if json_data["misbehaviour"] == True:
                                inputs.append([np.squeeze(image), json_data['expected_label']])
            for subdir, _, files in os.walk(dst3, followlinks=False):
                # Consider only the files that match the pattern
                    for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:    
                            json_path = svg_path.replace(".svg", ".json")            
                            with open(json_path) as jf:
                                json_data = json.load(jf)

                            npy_path = svg_path.replace(".svg", ".npy") 
                            image = np.load(npy_path)
                            if json_data["misbehaviour"] == True:
                                inputs.append([np.squeeze(image), json_data['expected_label']])
            
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
            target_y_test = keras.utils.to_categorical(target_y_test, NUM_CLASSES)
            target_y_train = keras.utils.to_categorical(target_y_train, NUM_CLASSES)

            target_y_train = np.concatenate((np.array(target_y_train), y_train), axis=0)
            target_x_train =  np.concatenate((np.array(target_x_train), x_train), axis=0)

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
                
