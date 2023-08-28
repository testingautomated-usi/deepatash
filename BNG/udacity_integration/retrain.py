
from pathlib import Path
import sys
import os
import argparse

path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

from random import shuffle
import tensorflow as tf
import json
import numpy as np
from config import DIVERSITY_METRIC, NUM_RETRAIN, model_file
from self_driving.road_bbox import RoadBoundingBox
from self_driving.decal_road import DecalRoad
from self_driving.beamng_member import BeamNGMember
from udacity_integration.train_dataset_recorder_brewer import run_sim
from udacity_integration.train_from_recordings import load_data_test, load_data_from_folder, s2b
from keras.optimizers import Adam
from udacity_integration.batch_generator import Generator
import self_driving.beamng_config as cfg
import self_driving.beamng_problem as BeamNGProblem

NUM_SPLINE_NODES = 20
bbox_size=(-250, 0, 250, 500)
road_bbox = RoadBoundingBox(bbox_size)
model_name = "self-driving-car-178-2020"

APPROACH = "nsga2"



def compute_success_rate(model_name, roads):
    success = 0
    config = cfg.BeamNGConfig()
    config.keras_model_file = model_name+".h5"
    problem = BeamNGProblem.BeamNGProblem(config)
    for road in roads:
        road.config = config
        road.problem = problem
        road.evaluate()

        border = road.distance_to_boundary
        if border > 0: 
            success +=1       
    return success/len(roads)


def retrain(folder_name, i, args):
    
    # Load the pre-trained model.
    from keras.models import load_model
    model = load_model(model_file)

    # load original test set
    x_test, y_test = load_data_test("test")
    test_generator = Generator(x_test, y_test, False, args)

    # load original train set
    x_train, y_train = load_data_test("train")

    # load target train set
    target_x_train, target_y_train = load_data_test(folder_name+"/train")

    # concate to the original training set
    target_y_train = np.concatenate((np.array(target_y_train), y_train), axis=0)
    target_x_train =  np.concatenate((np.array(target_x_train), x_train), axis=0)
    
    train_generator = Generator(target_x_train, target_y_train, True, args)

    # load target test set
    target_x_test, target_y_test = load_data_test(folder_name+"/test")    
    validation_generator = Generator(target_x_test, target_y_test, False, args)

    score = model.evaluate(test_generator, verbose=1)
    test_accuracy_before = score

    score = model.evaluate(validation_generator, verbose=1)
    test_accuracy_target_before = score

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    hist = model.fit(train_generator,
                        validation_data=validation_generator,
                        epochs=args.nb_epoch,
                        use_multiprocessing=False,
                        max_queue_size=10,
                        workers=4,
                        # callbacks=[checkpoint],
                        verbose=1)
    
    model.save(f'data/trained_models_colab/{folder_name}_{i}.h5')

    test_accuracy_target_after= hist.history['val_loss'][0]

    score = model.evaluate(test_generator, verbose=1)
    test_accuracy_after =  score


    return test_accuracy_before, test_accuracy_target_before, test_accuracy_target_after, test_accuracy_after


def MSE(feature_combinations, args):
    dst = "../experiments/data/bng/retrain/MSE"
    Path(dst).mkdir(parents=True, exist_ok=True)

    for features in feature_combinations:
        for i in range(6, NUM_RETRAIN+1):
            dst1 = f"../experiments/data/bng/DeepAtash-LR/target_cell_in_dark/{features}/{i}-{APPROACH}_-features_{features}-diversity_{DIVERSITY_METRIC}/output"
            dst2 = f"../experiments/data/bng/DeepAtash-LR/target_cell_in_grey/{features}/{i}-{APPROACH}_-features_{features}-diversity_{DIVERSITY_METRIC}/output"
            dst3 = f"../experiments/data/bng/DeepAtash-LR/target_cell_in_white/{features}/{i}-{APPROACH}_-features_{features}-diversity_{DIVERSITY_METRIC}/output"
            
            inputs = []
            for subdir, _, files in os.walk(dst1, followlinks=False):
                # Consider only the files that match the pattern
                for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    with open(json_path) as jf:
                        json_data = json.load(jf)
                
                    if json_data["misbehaviour"] == True:
                        print(".", end='', flush=True)  
                        inputs.append((json_data["control_nodes"], json_data["sample_nodes"]))

            for subdir, _, files in os.walk(dst2, followlinks=False):
                # Consider only the files that match the pattern
                for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    with open(json_path) as jf:
                        json_data = json.load(jf)
                
                    if json_data["misbehaviour"] == True:
                        print(".", end='', flush=True)  
                        inputs.append((json_data["control_nodes"], json_data["sample_nodes"]))
            
            for subdir, _, files in os.walk(dst3, followlinks=False):
                # Consider only the files that match the pattern
                for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    with open(json_path) as jf:
                        json_data = json.load(jf)
                
                    if json_data["misbehaviour"] == True:
                        print(".", end='', flush=True)  
                        inputs.append((json_data["control_nodes"], json_data["sample_nodes"]))
            
            folder_name =  str(i)+"-"+features
            print(f"{folder_name} num of inputs: {len(inputs)}")

            # split to train and test
            train_test_split = int(len(inputs) * 0.8)
            inputs_train = inputs[:train_test_split]
            inputs_test = inputs[train_test_split:]

            

            for control_nodes, sample_nodes in inputs_train:
                road = BeamNGMember(control_nodes, sample_nodes, NUM_SPLINE_NODES, road_bbox)
                street = DecalRoad('street_1', drivability=1, material='').add_4d_points(road.sample_nodes)
                run_sim(street, folder_name+"/train")


            for control_nodes, sample_nodes in inputs_test:
                road = BeamNGMember(control_nodes, sample_nodes, NUM_SPLINE_NODES, road_bbox)
                street = DecalRoad('street_1', drivability=1, material='').add_4d_points(road.sample_nodes)
                run_sim(street, folder_name+"/test")
                    
            
            if len(inputs) > 1:
                for rep in range(1, NUM_RETRAIN+1):
                    t0, t1, t2, t3 = retrain(folder_name, rep, args)

                    dict_report = {
                        "approach": "Before",
                        "features": features,
                        "MSE test set": t0,
                        "MSE target test set": t1
                    }
                    filedst = f"{dst}/report-{features}-before-{i}-{rep}.json"
                    with open(filedst, 'w') as f:
                        (json.dump(dict_report, f, sort_keys=True, indent=4))


                    dict_report = {
                        "approach": "After",
                        "features": features,
                        "MSE test set": t3,
                        "MSE target test set": t2
                    }
                    filedst = f"{dst}/report-{features}-after-{i}-{rep}.json"
                    with open(filedst, 'w') as f:
                        (json.dump(dict_report, f, sort_keys=True, indent=4))


def success_rate(feature_combinations, args):

    dst = "../experiments/data/bng/retrain/Success"
    Path(dst).mkdir(parents=True, exist_ok=True)

    # original test set
    test_dst = "data/retrain/test"
    inputs_original_test = []
    for subdir, _, files in os.walk(test_dst, followlinks=False):
        # Consider only the files that match the pattern
        for json_path in [os.path.join(subdir, f) for f in files if f.endswith(".json")]:
            with open(json_path) as jf:
                json_data = json.load(jf)
                print(".", end='', flush=True)  
                road = BeamNGMember(json_data["control_nodes"], json_data["sample_nodes"], NUM_SPLINE_NODES, road_bbox)
                inputs_original_test.append(road)

    s0 = compute_success_rate(model_name, inputs_original_test)

    for features in feature_combinations:
        for i in range(1, NUM_RETRAIN+1):
            dst1 = f"../experiments/data/bng/DeepAtash-LR/target_cell_in_dark/{features}/{i}-{APPROACH}_-features_{features}-diversity_{DIVERSITY_METRIC}/output"
            dst2 = f"../experiments/data/bng/DeepAtash-LR/target_cell_in_grey/{features}/{i}-{APPROACH}_-features_{features}-diversity_{DIVERSITY_METRIC}/output"
            dst3 = f"../experiments/data/bng/DeepAtash-LR/target_cell_in_white/{features}/{i}-{APPROACH}_-features_{features}-diversity_{DIVERSITY_METRIC}/output"
            
            inputs = []
            for subdir, _, files in os.walk(dst1, followlinks=False):
                # Consider only the files that match the pattern
                for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    with open(json_path) as jf:
                        json_data = json.load(jf)
                
                    if json_data["misbehaviour"] == True:
                        print(".", end='', flush=True)  
                        inputs.append((json_data["control_nodes"], json_data["sample_nodes"]))

            for subdir, _, files in os.walk(dst2, followlinks=False):
                # Consider only the files that match the pattern
                for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    with open(json_path) as jf:
                        json_data = json.load(jf)
                
                    if json_data["misbehaviour"] == True:
                        print(".", end='', flush=True)  
                        inputs.append((json_data["control_nodes"], json_data["sample_nodes"]))
            
            for subdir, _, files in os.walk(dst3, followlinks=False):
                # Consider only the files that match the pattern
                for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                    with open(json_path) as jf:
                        json_data = json.load(jf)
                
                    if json_data["misbehaviour"] == True:
                        print(".", end='', flush=True)  
                        inputs.append((json_data["control_nodes"], json_data["sample_nodes"]))
            
            folder_name =  str(i)+"-"+features
            print(f"{folder_name} num of inputs: {len(inputs)}")

            # split to train and test
            train_test_split = int(len(inputs) * 0.8)
            inputs_test = inputs[train_test_split:]

            roads_test = []
            
            for control_nodes, sample_nodes in inputs_test:
                road = BeamNGMember(control_nodes, sample_nodes, NUM_SPLINE_NODES, road_bbox)
                roads_test.append(road)
                    
            
            if len(inputs) > 1:
                for rep in range(1, NUM_RETRAIN+1):
                    s2 = compute_success_rate(f"{folder_name}_{i}", inputs_original_test)
                    s3 = compute_success_rate(f"{folder_name}_{i}", roads_test)
                    dict_report = {
                        "approach": "Before",
                        "features": features,
                        "success test set": s0,
                        "success target test set": 0
                    }
                    filedst = f"{dst}/report-{features}-before-{i}-{rep}.json"
                    with open(filedst, 'w') as f:
                        (json.dump(dict_report, f, sort_keys=True, indent=4))


                    dict_report = {
                        "approach": "After",
                        "features": features,
                        "success test set": s2,
                        "success target test set": s3
                    }
                    filedst = f"{dst}/report-{features}-after-{i}-{rep}.json"
                    with open(filedst, 'w') as f:
                        (json.dump(dict_report, f, sort_keys=True, indent=4))
                


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-m', help='metric', dest='metric', type=str, default='.')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='.')
    # parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=20)
    # parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=100)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=128)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    args = parser.parse_args()

    feature_combinations = ["Curvature-MeanLateralPosition", "Curvature-SegmentCount", "SegmentCount-SDSteeringAngle"] #, "Curvature-MeanLateralPosition", ] # 

    if args.metric  == "MSE":
        MSE(feature_combinations, args)
    elif args.metric == "Success":
        success_rate(feature_combinations, args)