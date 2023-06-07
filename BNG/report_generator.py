
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
from pathlib import Path
import time
import matplotlib
from self_driving.catmull_rom import catmull_rom
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

import utils as us
from sample import Sample
from feature import Feature
from self_driving.simulation_data import SimulationParams, SimulationData, SimulationDataRecord, SimulationInfo
from self_driving.road_bbox import RoadBoundingBox
from self_driving.vehicle_state_reader import VehicleState
from self_driving.beamng_member import BeamNGMember
from self_driving.decal_road import DecalRoad
import self_driving.beamng_problem as BeamNGProblem
import self_driving.beamng_individual as BeamNGIndividual
import self_driving.beamng_config as cfg
from config import TARGET_THRESHOLD, META_FILE, TARGET_SIZE
from typing import Tuple, List
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_data(data: np.ndarray, n_clusters_interval: Tuple[int, int]) -> Tuple[List[int], List[float]]:
    """
    :param data: data to cluster
    :param n_clusters_interval: (min number of clusters, max number of clusters) for silhouette analysis
    :return: list of labels, list of centroid coordinates, optimal silhouette score
    """
    assert n_clusters_interval[0] >= 2, 'Min number of clusters must be >= 2'
    range_n_clusters = np.arange(n_clusters_interval[0], n_clusters_interval[1])
    optimal_score = -1
    optimal_n_clusters = -1
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(data)
        try:
            silhouette_avg = silhouette_score(data, cluster_labels)  # throws ValueError
            print("For n_clusters = {}, the average silhouette score is: {}".format(n_clusters, silhouette_avg))
            if silhouette_avg > optimal_score:
                optimal_score = silhouette_avg
                optimal_n_clusters = n_clusters
        except Exception as ex:
            optimal_n_clusters = 1

    assert optimal_n_clusters != -1, 'Error in silhouette analysis'
    print('Best score is {} for n_cluster = {}'.format(optimal_score, optimal_n_clusters))

    clusterer = KMeans(n_clusters=optimal_n_clusters).fit(data)
    return clusterer.labels_, clusterer.cluster_centers_


def load_data_all(dst, features):
    """
    :param dst: source folder for DeepAtash experiments
    :param features: feature combinations to consider
    :return: list of misbehaviours in the archives of DeepAtash with different configurations
    """
    inputs = []
    inputs_target = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]: 
             if features in json_path:     
                print(".", end='', flush=True)  

                if "ga_" in json_path:
                    y2 = "GA"
                elif "nsga2" in json_path:
                    y2 = "NSGA2"

                y1 = "INPUT"
        
                with open(json_path) as jf:
                    json_data = json.load(jf)
                
                if json_data["misbehaviour"] == True:
                    inputs.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"]), json_data["elapsed"]])
                    if float(json_data["distance to target"]) == 0:
                        inputs_target.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"]), json_data["elapsed"]])
    print(len(inputs))                            
    return inputs, inputs_target


def load_data(dst, i, approach, div):
    """
    :param dst: source folder for DeepAtash experiments
    :param i: run number
    :param approach: ga or nsga2
    :param div: input, latent or heatmpa
    :return: list of misbehaviours in the archives of DeepAtash with specified configurations
    """
    inputs = []
    inputs_target = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if i+approach in subdir and div in subdir and "output" in subdir:
            for json_path in [os.path.join(subdir, f) for f in files if f.endswith("road.json")]:
                print(".", end='', flush=True)  

                if "ga_" in json_path:
                    y2 = "GA"
                elif "nsga2" in json_path:
                    y2 = "NSGA2"

                y1 = "INPUT"
        
                with open(json_path) as jf:
                    json_data = json.load(jf)

                if json_data["misbehaviour"] == True:
                    inputs.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"]), json_data["elapsed"]])
                    if float(json_data["distance to target"]) == 0:
                        inputs_target.append([json_data["sample_nodes"], f"{y2}-{y1}", json_data['misbehaviour'], float(json_data["distance to target"]), json_data["elapsed"]])
                    
    return inputs, inputs_target


def plot_tSNE(inputs, _folder, features, div, ii=0):
    """
    This function computes diversity using t-sne
    :param inputs: list of inputs for t-SNE
    :param _folder: destination folder to save the plot
    :param features: feature combination
    :param div: corresponding sparseness metric (input, latent or heatmap)
    :param ii: run number
    """
    X, y = [], []
    for i in inputs:
        X.append(list(matplotlib.cbook.flatten(i[0])))
        y.append(i[1])

    
    X = np.array(X)

    feat_cols = ['points'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=1, n_iter=3000)
    tsne_results = tsne.fit_transform(df[feat_cols].values)
   
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]
    
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        # palette=sns.color_palette("hls", n_colors=10),
        data=df,
        legend="full",
        alpha=0.3
    )
    fig.savefig(f"{_folder}/tsne-diag-{features}-{div}-{ii}-1.pdf", format='pdf')

    return df


def compute_tSNE_and_cluster_all(inputs, targets, _folder, features, div, ii=0):

    target_input_ga = 0
    target_input_nsga2 = 0

    div_input_ga = 0
    div_input_nsga2 = 0

    input_ga = 0
    input_nsga2 = 0

    div_target_input_ga = 0
    div_target_input_nsga2 = 0

    for i in targets:
        if i[1] == "GA-INPUT":
            target_input_ga += 1
        if i[1] == "NSGA2-INPUT":
            target_input_nsga2 += 1

    if len(inputs) > 3:

        df = plot_tSNE(inputs, _folder, features, div, ii)    
        df = df.reset_index()  # make sure indexes pair with number of rows
    

        np_data_cols = df.iloc[:,403:405]

        n = len(inputs) - 1
        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-1.pdf", format='pdf')
        
        df_nsga2 = df[df.label == "NSGA2-INPUT"]
        df_ga = df[df.label =="GA-INPUT"]

        num_clusters = len(centers)

        div_input_ga = df_ga.nunique()['Clusters']/num_clusters
        div_input_nsga2 = df_nsga2.nunique()['Clusters']/num_clusters
        
        input_ga = len(df_ga.index)
        input_nsga2 = len(df_nsga2.index)

    else:
        for i in inputs:
            if i[1] == "GA-INPUT":
                input_ga += 1
                div_input_ga = 1.0
            if i[1] == "NSGA2-INPUT":
                input_nsga2 += 1
                div_input_nsga2 = 1.0

    if len(targets) > 3:

        df = plot_tSNE(targets, _folder, features, div, ii)    
        df = df.reset_index()  # make sure indexes pair with number of rows
        
        np_data_cols = df.iloc[:,403:405]

        n = len(targets) - 1
        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')
        
        df_target_nsga2 = df[df.label == "NSGA2-INPUT"]
        df_target_ga = df[df.label =="GA-INPUT"]

        num_clusters = len(centers)

        div_target_input_ga = df_target_ga.nunique()['Clusters']/num_clusters
        div_target_input_nsga2 = df_target_nsga2.nunique()['Clusters']/num_clusters
        
    else:
        for i in targets:
            if i[1] == "GA-INPUT":
                div_input_ga = 1.0
            if i[1] == "NSGA2-INPUT":
                div_input_nsga2 = 1.0    
        
    list_data = [("GA", "Input", div_input_ga, input_ga, target_input_ga, div_target_input_ga),\
             ("NSGA2", "Input", div_input_nsga2, input_nsga2, target_input_nsga2, div_target_input_nsga2)]        
                   
    return list_data


def find_best_div_approach(dst, feature_combinations):

    evaluation_area = ["target_cell_in_grey"] # "target_cell_in_dark","target_cell_in_grey"


    for evaluate in evaluation_area:
        
        for features in feature_combinations:
            for i in range(1, 11):
                inputs = []
                targets = []
                for subdir, _, _ in os.walk(f"{dst}/{evaluate}/{features}", followlinks=False):
                    if features in subdir and str(i)+"-" in subdir and evaluate in subdir and "output" in subdir and "sim_" not in subdir:
                        data_folder = subdir
                        all_inputs, all_targets = load_data_all(data_folder, features)
                        inputs = inputs + all_inputs
                        targets = targets + all_targets

                list_data = compute_tSNE_and_cluster_all(inputs, targets, f"{dst}/{evaluate}/{features}", features, i)

                
                for data in list_data:
                    dict_data = {
                        "approach": data[0],
                        "diversity": data[1],
                        "run": i,
                        "test input count": data[3],
                        "features": features,
                        "num tsne clusters": str(data[2]),
                        "test input count in target": data[4],
                        "target num tsne clusters": str(data[5])
                    }

                    filedest = f"{dst}/{evaluate}/{features}/report_{data[0]}-{data[1]}_{i}.json"
                    with open(filedest, 'w') as f:
                        (json.dump(dict_data, f, sort_keys=True, indent=4))


def generate_features(FEATURES):

    features = []
    with open(META_FILE, 'r') as f:
        meta = json.load(f)["features"]
        
    if "MeanLateralPosition" in FEATURES:
        f3 = Feature("mean_lateral_position", meta["mean_lateral_position"]["min"], meta["mean_lateral_position"]["max"], "mean_lateral_position", 25)
        features.append(f3)
    if "Curvature" in FEATURES:
        f1 = Feature("curvature",meta["curvature"]["min"], meta["curvature"]["max"], "curvature", 25)
        features.append(f1)
    if "SegmentCount" in FEATURES:
        f2 = Feature("segment_count",meta["segment_count"]["min"], meta["segment_count"]["max"], "segment_count", int(meta["segment_count"]["max"])+1)
        features.append(f2)
    if "SDSteeringAngle" in FEATURES:
        f1 = Feature("sd_steering_angle",meta["sd_steering_angle"]["min"], meta["sd_steering_angle"]["max"], "sd_steering", 25)
        features.append(f1)
    return features
       

def compute_targets_for_dh(dst, goal, features, metric):
    fts = generate_features(features)
    count = 0
    samples = []
    archive_samples = []
    _config = cfg.BeamNGConfig()
    _config.name = ""

    for subdir, _, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("simulation") and
                            f.endswith(".json")
                        )]:

            with open(json_path) as jf:
                data = json.load(jf)
            
            for info_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("info") and
                            f.endswith(".json")
                        )]:

                with open(info_path) as ff:
                    info_data = json.load(ff)            
                            
                nodes = data["road"]["nodes"]
                bbox_size=(-250, 0, 250, 500)
                member = BeamNGMember(nodes, nodes, 20, RoadBoundingBox(bbox_size))
                records = data["records"]
                simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                sim_name = simulation_id # member.config.simulation_name.replace('$(id)', simulation_id)
                simulation_data = SimulationData(sim_name)
                states = []
                for record in records:
                    state = VehicleState(timer=record["timer"]
                                        , damage=record["damage"]
                                        , pos=record["pos"]
                                        , dir=record["dir"]
                                        , vel=record["vel"]
                                        , gforces=record["gforces"]
                                        , gforces2=record["gforces2"]
                                        , steering=record["steering"]
                                        , steering_input=record["steering_input"]
                                        , brake=record["brake"]
                                        , brake_input=record["brake_input"]
                                        , throttle=record["throttle"]
                                        , throttle_input=record["throttle_input"]
                                        , throttleFactor=record["throttleFactor"]
                                        , engineThrottle=record["engineThrottle"]
                                        , wheelspeed=record["engineThrottle"]
                                        , vel_kmh=record["engineThrottle"])

                    sim_data_record = SimulationDataRecord(**state._asdict(),
                                                        is_oob=record["is_oob"],
                                                        oob_counter=record["oob_counter"],
                                                        max_oob_percentage=record["max_oob_percentage"],
                                                        oob_distance=record["oob_distance"])
                    states.append(sim_data_record)

                simulation_data.params = SimulationParams(beamng_steps=data["params"]["beamng_steps"], delay_msec=int(data["params"]["delay_msec"]))
                simulation_data.control_nodes = nodes
                simulation_data.road = DecalRoad.from_dict(data["road"])
                simulation_data.info = SimulationInfo()
                simulation_data.info.start_time = data["info"]["start_time"]
                simulation_data.info.end_time = data["info"]["end_time"]
                simulation_data.info.elapsed_time = data["info"]["elapsed_time"]
                simulation_data.info.success = data["info"]["success"]
                simulation_data.info.computer_name = data["info"]["computer_name"]
                simulation_data.info.id = data["info"]["id"]

                simulation_data.states = states

                if len(states) > 0:
                    member.distance_to_boundary = simulation_data.min_oob_distance()
                    member.simulation = simulation_data
                    member.sample_nodes = nodes # catmull_rom(member.control_nodes, member.num_spline_nodes)
                    misbehaviour = states[-1].is_oob
                    oob_ff = states[-1].oob_distance
                else:
                    misbehaviour = False
                    break
                
                ind = BeamNGIndividual.BeamNGIndividual(member, _config)
                ind.oob_ff = oob_ff
                sample = Sample(ind)
                #us.is_oob(sample.ind.sample_nodes, sample.ind.member.simulation.states)
                
                sample.misbehaviour = misbehaviour

                sample.elapsed = "00:00:0.0"

                sample.features["curvature"] = info_data["features"]["curvature"]
                sample.features["segment_count"] = info_data["features"]["segment_count"]
                sample.features["sd_steering_angle"] = info_data["features"]["sd_steering"]


                b = tuple()
                for ft in fts:
                    i = ft.get_coordinate_for(sample)
                    if i != None:
                        b = b + (i,)
                    else:
                        b = np.inf
                
                sample.distance_to_target = us.manhattan(b, goal)

                if sample.distance_to_target <= TARGET_THRESHOLD:
                    samples.append(sample)
                    count += 1


    archive = []
    for sample in samples:
        if len(archive) == 0:
            sample.sparseness = np.inf
            archive.append(sample)
        else:
            dmin = np.inf
            for idx in range(len(archive)):
                a = archive[idx]
                dist = us.get_distance_by_metric(a, sample, metric)
                if dist < dmin:
                    dmin = dist
                    idx_min = idx

            sample.sparseness = dmin
            if len(archive)/TARGET_SIZE < 1:               
                if dmin > 0:                  
                    archive.append(sample)
                    archive[idx_min].sparseness = dmin
            else:
                c = sorted(archive, key=lambda x: (x.distance_to_target, -x.sparseness), reverse=True)[0]
                if c.distance_to_target > sample.distance_to_target:
                    archive.append(sample)
                    archive[idx_min].sparseness = dmin
                elif c.distance_to_target == sample.distance_to_target:
                    # ind has better performance
                    if sample.ind.oob_ff < c.ind.oob_ff:
                        archive.append(sample)
                        archive[idx_min].sparseness = dmin
                    # c and ind have the same performance
                    elif sample.ind.oob_ff == c.ind.oob_ff:
                        # ind has better sparseness                        
                        if dmin > c.sparseness:
                            archive.append(sample)
                            archive[idx_min].sparseness = dmin

    target_samples = []
    for sample in archive:
        if sample.misbehaviour == True:
            print(".", end='', flush=True)
            archive_samples.append([sample.ind.m.sample_nodes, f"DeepHyperion", sample.misbehaviour, sample.distance_to_target, sample.elapsed])
            if sample.distance_to_target == 0:
                target_samples.append([sample.ind.m.sample_nodes, f"DeepHyperion", sample.misbehaviour, sample.distance_to_target, sample.elapsed])


    print("DeepHyperion", features, len(archive_samples))
    return archive_samples, target_samples


def elapsed_to_millisec(elapsed):
    # compute milli seconds for sample's elapsed time
    times = re.split(r"[:.]", elapsed)
    millisecs = float(times[0])*3600+float(times[1])*60+float(times[2])+float(times[3])/1000000.
    return millisecs


def compute_metrics(inputs_da, targets_da, inputs_dh, targets_dh, _folder, features, approach, div, ii=0, num=10):
    div_da = 0
    div_dh = 0
    div_targets_da = 0
    div_targets_dh = 0
    # compute area under the curve for performance
    input_data = [0]
    time_data = [0]
    current = 0
    for sample in sorted(inputs_da,key=lambda s: elapsed_to_millisec(s[4]),reverse=False):
        input_data.append(current+1)
        millisecs = elapsed_to_millisec(sample[4])
        time_data.append(millisecs)

    auc_deepatash = np.trapz(x = time_data, y= input_data)

    input_data = [0]
    time_data = [0]
    current = 0
    for sample in sorted(inputs_dh,key=lambda s: elapsed_to_millisec(s[4]),reverse=False):
        input_data.append(current+1)
        millisecs = elapsed_to_millisec(sample[4])
        time_data.append(millisecs)
    
    auc_deephyperion = np.trapz(x = time_data, y= input_data)
    
    inputs = inputs_da + inputs_dh


    if len(inputs) > 3:
        df = plot_tSNE(inputs, _folder, features, div, ii)

        df = df.reset_index()  # make sure indexes pair with number of rows
        np_data_cols = df.iloc[:,403:405]

        n = len(inputs) - 1
        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-1.pdf", format='pdf')
        
        df_da = df[df.label == f"{approach}-{div}"]
        df_dh = df[df.label =="DeepHyperion"]

        num_clusters = len(centers)

        div_da = df_da.nunique()['Clusters']/num_clusters
        div_dh = df_dh.nunique()['Clusters']/num_clusters

    else:
        for i in inputs:
            if i[1] == f"{approach}-{div}":
                div_da = 1.0
            if i[1] == "DeepHyperion":
                div_dh = 1.0
    
    targets = targets_da + targets_dh
    if len(targets) > 3:
        df = plot_tSNE(targets, _folder, features, div, ii)

        df = df.reset_index()  # make sure indexes pair with number of rows
        np_data_cols = df.iloc[:,403:405]


        n = len(inputs) - 1
        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-1.pdf", format='pdf')
        
        df_da = df[df.label == f"{approach}-{div}"]
        df_dh = df[df.label =="DeepHyperion"]

        num_clusters = len(centers)

        div_targets_da = df_da.nunique()['Clusters']/num_clusters
        div_targets_dh = df_dh.nunique()['Clusters']/num_clusters

    else:
        for i in targets:
            if i[1] == f"{approach}-{div}":
                div_targets_da = 1.0
            if i[1] == "DeepHyperion":
                div_targets_dh = 1.0   

    list_data = [("DeepAtash", div_da , len(inputs_da), div_targets_da, len(targets_da), auc_deepatash),
            ("DeepHyperion", div_dh, len(inputs_dh), div_targets_dh, len(targets_dh), auc_deephyperion)]
 
    return list_data


def compare_with_dh(approach, div, features, target_area):

    result_folder = f"../experiments/data/bng/comparison/{target_area}"
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    
    for feature in features:

        dst = f"../experiments/data/bng/DeepAtash/{target_area}/{feature[0]}"
        dst_dh = f"../experiments/data/bng/DeepHyperion/{feature[0]}"


        for i in range(1, 11):
            for subdir, _, _ in os.walk(dst_dh, followlinks=False):
                if "dh-"+str(i) in subdir and "beamng_nvidia_runner" in subdir:
                    inputs_dh, targets_dh = compute_targets_for_dh(subdir, feature[1], feature[0], div)
                    break
            
            inputs_focused, targets_focused = load_data(dst, str(i)+"-", approach, div)

            if "ga" == approach:
                approach2 = "GA"
            elif "nsga2" == approach:
                approach2 = "NSGA2"

            list_data = compute_metrics(inputs_focused, targets_focused, inputs_dh, targets_dh, result_folder, feature[0], approach2, div, str(i))
           

            for data in list_data:
                dict_data = {
                    "approach": data[0],
                    "run": i,
                    "test input count": data[2],
                    "features": feature[0],
                    "num tsne clusters": str(data[1]),
                    "target num tsne clusters": str(data[3]),
                    "test input count in target": data[4],
                    "auc": str(data[5])
                }

                filedest = f"{result_folder}/report_{data[0]}_{feature[0]}_{target_area}_{i}.json"
                with open(filedest, 'w') as f:
                    (json.dump(dict_data, f, sort_keys=True, indent=4))



if __name__ == "__main__": 
    if sys.argv[1] == "dark": 
        features = [ ("Curvature-SegmentCount", (22, 6))]
        compare_with_dh("ga", "INPUT", features, "target_cell_in_dark")
    elif sys.argv[1] == "grey": 
        features = [("SegmentCount-SDSteeringAngle",(7, 17))]
        compare_with_dh("ga", "INPUT", features, "target_cell_in_grey")
    elif sys.argv[1] == "ga":
        features = [("Curvature-SegmentCount",(22, 5))]
        compare_with_dh("ga", "INPUT", features, "target_cell_in_white")
    else:
        dst = "../experiments/data/bng/DeepAtash"
        feature_combinations = ["SegmentCount-SDSteeringAngle"]
        find_best_div_approach(dst, feature_combinations)


