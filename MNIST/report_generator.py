
from ast import Lambda
import os
import io
import json
import csv
from re import sub
from xml.sax.xmlreader import InputSource
import tensorflow as tf
from evaluator import Evaluator
from config import BITMAP_THRESHOLD, EXPECTED_LABEL, MODEL, move_range, orientation_range, bitmaps_range, TARGET_THRESHOLD
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import utils as us
from scipy import stats
from matplotlib.pyplot import boxplot
import statsmodels.stats.power as pw
from sample import Sample
import random
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List

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
        silhouette_avg = silhouette_score(data, cluster_labels)  # throws ValueError
        print("For n_clusters = {}, the average silhouette score is: {}".format(n_clusters, silhouette_avg))
        if silhouette_avg > optimal_score:
            optimal_score = silhouette_avg
            optimal_n_clusters = n_clusters

    assert optimal_n_clusters != -1, 'Error in silhouette analysis'
    print('Best score is {} for n_cluster = {}'.format(optimal_score, optimal_n_clusters))

    clusterer = KMeans(n_clusters=optimal_n_clusters).fit(data)
    return clusterer.labels_, clusterer.cluster_centers_


model = tf.keras.models.load_model(MODEL)
encoder = tf.keras.models.load_model("models/vae_encoder_test", compile=False)
evaluator = Evaluator()

num_cells = 100

def load_data_all(dst, features):
    inputs = []
    inputs_in_target = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:
            if features in svg_path:  
                print(".", end='', flush=True)  

                if "ga_" in svg_path:
                    y2 = "GA"
                elif "nsga2" in svg_path:
                    y2 = "NSGA2"

                if "LATENT" in svg_path:
                    y1 = "LATENT"
                elif "INPUT" in svg_path:
                    y1 = "INPUT"
                elif "HEATMAP" in svg_path:
                    y1 = "HEATMAP"

                with open(svg_path, 'r') as input_file:
                    xml_desc = input_file.read()       
                            
                json_path = svg_path.replace(".svg", ".json")            
                with open(json_path) as jf:
                    json_data = json.load(jf)

                npy_path = svg_path.replace(".svg", ".npy") 
                image = np.load(npy_path)

                png_path = svg_path.replace(".svg", ".png")

                inputs.append([image, f"{y2}-{y1}", json_data['predicted_label'], float(json_data["distance to target"])])
   
    return inputs


def load_data(dst, i, approach, div):
    inputs = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if i+approach in subdir and div in subdir:
            for svg_path in [os.path.join(subdir, f) for f in files if f.endswith(".svg")]:  
                    print(".", end='', flush=True)  

                    if "ga_" in svg_path:
                        y2 = "GA"
                    elif "nsga2" in svg_path:
                        y2 = "NSGA2"

                    if "LATENT" in svg_path:
                        y1 = "LATENT"
                    elif "INPUT" in svg_path:
                        y1 = "INPUT"
                    elif "HEATMAP" in svg_path:
                        y1 = "HEATMAP"

                    with open(svg_path, 'r') as input_file:
                        xml_desc = input_file.read()       
                                
                    json_path = svg_path.replace(".svg", ".json")            
                    with open(json_path) as jf:
                        json_data = json.load(jf)

                    npy_path = svg_path.replace(".svg", ".npy") 
                    image = np.load(npy_path)

                    inputs.append([image, f"{y2}-{y1}", json_data['predicted_label'], float(json_data["distance to target"])])

    return inputs


def plot_tSNE(inputs, _folder, features, div, ii=0):
    """
    This function computes diversity using t-sne
    """
    X, y, imgs = [], [], []
    for i in inputs:
        X.append(i[0].flatten())
        y.append(i[1])
        imgs.append(i[2])
    
    X = np.array(X)

    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X,columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    tsne = TSNE(n_components=2, verbose=1, perplexity=0.1, n_iter=3000)
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
    fig.savefig(f"{_folder}/tsne-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')

    return df


def compute_tSNE_and_cluster_all(inputs,  _folder, features, div, ii=0, num=10):

    target_input_ga = 0
    target_input_nsga2 = 0

    target_latent_ga = 0
    target_latent_nsga2 = 0

    target_heatmap_ga = 0
    target_heatmap_nsga2 = 0

    input_ga = 0
    input_nsga2 = 0

    latent_ga = 0
    latent_nsga2 = 0

    heatmap_ga = 0
    heatmap_nsga2 = 0


    for i in inputs:
        if i[3] == 0.0:
            if i[1] == "GA-INPUT":
                target_input_ga += 1
            if i[1] == "NSGA2-INPUT":
                target_input_nsga2 += 1
            if i[1] == "GA-LATENT":
                target_latent_ga += 1
            if i[1] == "NSGA2-LATENT":
                target_latent_nsga2 += 1
            if i[1] == "GA-HEATMAP":
                target_heatmap_ga += 1
            if i[1] == "NSGA2-HEATMAP":
                target_heatmap_nsga2 += 1

    if len(inputs) > 3:

        df = plot_tSNE(inputs, _folder, features, div, ii)
        df = df.reset_index()  # make sure indexes pair with number of rows


        np_data_cols = df.iloc[:,787:789]

        n = len(inputs) - 1

        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')

        
        df_nsga2_input = df[df.label == "NSGA2-INPUT"]
        df_ga_input = df[df.label =="GA-INPUT"]

        df_nsga2_latent = df[df.label == "NSGA2-LATENT"]
        df_ga_latent = df[df.label =="GA-LATENT"]

        df_nsga2_heatmap = df[df.label == "NSGA2-HEATMAP"]
        df_ga_heatmap = df[df.label =="GA-HEATMAP"]

        num_clusters = len(centers)

        div_input_ga = df_ga_input.nunique()['Clusters']/num_clusters
        div_input_nsga2 = df_nsga2_input.nunique()['Clusters']/num_clusters

        div_latent_ga = df_ga_latent.nunique()['Clusters']/num_clusters
        div_latent_nsga2 = df_nsga2_latent.nunique()['Clusters']/num_clusters

        div_heatmap_ga = df_ga_heatmap.nunique()['Clusters']/num_clusters
        div_heatmap_nsga2 = df_nsga2_heatmap.nunique()['Clusters']//num_clusters

        list_data = [("GA", "Input", div_input_ga,  len(df_ga_input.index), target_input_ga), ("NSGA2", "Input", div_input_nsga2,  len(df_nsga2_input.index), target_input_nsga2), \
                        ("GA", "Latent", div_latent_ga,  len(df_ga_latent.index), target_latent_ga), ("NSGA2", "Latent", div_latent_nsga2,  len(df_nsga2_latent.index), target_latent_nsga2), \
                        ("GA", "Heatmap",  div_heatmap_ga,  len(df_ga_heatmap.index), target_heatmap_ga), ("NSGA2", "Heatmap", div_heatmap_nsga2,  len(df_nsga2_heatmap.index), target_heatmap_nsga2)]
    else:
        for i in inputs:
            if i[1] == "GA-INPUT":
                input_ga += 1
            if i[1] == "NSGA2-INPUT":
                input_nsga2 += 1
            if i[1] == "GA-LATENT":
                latent_ga += 1
            if i[1] == "NSGA2-LATENT":
                latent_nsga2 += 1
            if i[1] == "GA-HEATMAP":
                heatmap_ga += 1
            if i[1] == "NSGA2-HEATMAP":
                heatmap_nsga2 += 1   

        list_data = [("GA", "Input",input_ga,  input_ga, target_input_ga), ("NSGA2", "Input", input_nsga2,  input_nsga2, target_input_nsga2), \
                        ("GA", "Latent", latent_ga, latent_ga, target_latent_ga), ("NSGA2", "Latent", latent_nsga2,  latent_nsga2, target_latent_nsga2), \
                        ("GA", "Heatmap",  heatmap_ga,  heatmap_ga, target_heatmap_ga), ("NSGA2", "Heatmap", heatmap_nsga2,  heatmap_nsga2, target_heatmap_nsga2)]
    

    return list_data


def find_best_div_approach(dst, feature_combinations):

    evaluation_area = ["target_cell_in_white"] #] # "target_cell_in_dark", "target_cell_in_gray",

    for evaluate in evaluation_area:
        
        for features in feature_combinations:
            for i in range(1, 11):
                inputs = []
                for subdir, dirs, files in os.walk(dst, followlinks=False):
                    if features in subdir and str(i)+"-" in subdir and evaluate in subdir:
                        data_folder = subdir
                        all_inputs = load_data_all(data_folder, features)
                        inputs = inputs + all_inputs


                list_data = compute_tSNE_and_cluster_all(inputs, f"{dst}/{evaluate}/{features}", features, i)
                
                for data in list_data:
                    dict_data = {
                        "approach": data[0],
                        "diversity": data[1],
                        "run": i,
                        "test input count": data[3],
                        "features": features,
                        "num tsne clusters": str(data[2]),
                        "test input count in target": data[4]
                    }

                    filedest = f"{dst}/{evaluate}/{features}/report_{data[0]}-{data[1]}_{i}.json"
                    with open(filedest, 'w') as f:
                        (json.dump(dict_data, f, sort_keys=True, indent=4))


def compute_targets_for_dh(dst, goal, features, threshold, metric):
    count = 0
    samples = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("mbr") and
                            f.endswith(".json")
                        )]:
            with open(json_path) as jf:
                ind = json.load(jf)

            npy_path = json_path.replace(".json", ".npy")
            img = np.load(npy_path)

            svg_path = json_path.replace(".json", ".svg")
            with open(svg_path, 'r') as input_file:
                xml_desc = input_file.read()
        

            sample = Sample(xml_desc, EXPECTED_LABEL, int(ind["seed"]))

            sample.predicted_label = ind["predicted_label"]

            if features == "Moves-Bitmaps":
                cell = (ind["features"]["moves"]/move_range, ind["features"]["bitmaps"]/bitmaps_range)

            if features == "Orientation-Bitmaps":
                cell = (ind["features"]["orientation"]/orientation_range, ind["features"]["bitmaps"]/bitmaps_range)

            if features == "Orientation-Moves":
                cell = (ind["features"]["moves"]/move_range, ind["features"]["orientation"]/orientation_range)
            
            sample.distance_to_target = us.manhattan(cell, goal)
            if sample.distance_to_target <= TARGET_THRESHOLD and sample.is_misbehavior() == True:
                sample.compute_explanation()
                sample.compute_latent_vector(encoder)
                samples.append(sample)
                count += 1

    samples = sorted(samples, key=lambda x: x.distance_to_target)
    archive = []
    for sample in samples:
        if len(archive) == 0:
            archive.append(sample)
        elif all(us.get_distance_by_metric(a, sample, metric)> threshold for a in archive):
            archive.append(sample)

    target_samples = []
    for sample in archive:
        print(".", end='', flush=True)
        target_samples.append([sample.purified, f"DeepHyperion", sample.predicted_label, sample.distance_to_target])

    print("DeepHyperion", features,len(target_samples))
    return target_samples


def compute_targets_for_dlf(dst, goal, features, threshold, metric):
    count = 0
    samples = []
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("mbr") and
                            f.endswith(".json")
                        )]:
            with open(json_path) as jf:
                ind = json.load(jf)

            npy_path = json_path.replace(".json", ".npy")
            img = np.load(npy_path)

            xml_desc = (ind["xml_desc"][2:len(ind["xml_desc"])-1]).replace("<?xml version=\\'1.0\\' encoding=\\'utf8\\'?>\\n", "")
            
            
            sample = Sample(xml_desc, EXPECTED_LABEL, int(ind["seed"]))
            sample.purified = img
            evaluator.evaluate(sample, model)


            bitmaps = us.bitmap_count(sample, BITMAP_THRESHOLD)
            moves = us.move_distance(sample)
            orientation = us.orientation_calc(sample, 0)

            if features == "Moves-Bitmaps":
                cell = (moves/move_range, bitmaps/bitmaps_range)

            if features == "Orientation-Bitmaps":
                cell = (orientation/orientation_range, bitmaps/bitmaps_range)

            if features == "Orientation-Moves":
                cell = (moves/move_range, orientation/orientation_range)
            
            sample.distance_to_target = us.manhattan(cell, goal)
            if sample.distance_to_target <= TARGET_THRESHOLD and sample.is_misbehavior() == True:
                sample.compute_explanation()
                sample.compute_latent_vector(encoder)
                samples.append(sample)
                count += 1

    samples = sorted(samples, key=lambda x: x.distance_to_target)
    archive = []
    for sample in samples:
        if len(archive) == 0:
            archive.append(sample)
        elif all(us.get_distance_by_metric(a, sample, metric)> threshold for a in archive):
            archive.append(sample)

    target_samples = []
    for sample in archive:
        print(".", end='', flush=True)
        target_samples.append([sample.purified, f"DLFuzz", sample.predicted_label, sample.distance_to_target])

    print("DLFuzz", features, len(target_samples))
    return target_samples


def compute_tSNE_cluster_vs_dh_and_dlf(inputs, _folder, features, approach, div, ii=0, num=10):

    target_deepatash = 0
    target_deephyperion = 0
    target_dlfuzz = 0

    deepatash = 0
    deephyperion = 0
    dlfuzz = 0

    dist_deepatash = 0.0
    dist_deephyperion = 0.0
    dist_dlfuzz = 0.0

    for i in inputs:
        if i[3] == 0.0:
            if i[1] == "NSGA2-LATENT":
                target_deepatash += 1
            if i[1] == "DLFuzz":
                target_dlfuzz += 1
            if i[1] == "DeepHyperion":
                target_deephyperion += 1

    if len(inputs) > 3:

        df = plot_tSNE(inputs, _folder, features, div, ii)
        df = df.reset_index()  # make sure indexes pair with number of rows

        np_data_cols = df.iloc[:,787:789]

        n = len(inputs) - 1

        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-0.1.pdf", format='pdf')

        
        df_da = df[df.label == "NSGA2-LATENT"]
        df_dh = df[df.label =="DeepHyperion"]
        df_dlf = df[df.label =="DLFuzz"]



        num_clusters = len(centers)

        div_da = df_da.nunique()['Clusters']/num_clusters
        div_dh = df_dh.nunique()['Clusters']/num_clusters
        div_dlf = df_dlf.nunique()['Clusters']/num_clusters

        list_data = [("DeepAtash", div_da, len(df_da.index), target_deepatash),
                    ("DLFuzz", div_dlf, len(df_dlf.index), target_dlfuzz),
                    ("DeepHyperion", div_dh, len(df_dh.index), target_deephyperion)]
    
    else:
        for i in inputs:
            if i[1] == "NSGA2-LATENT":
                deepatash += 1
                dist_deepatash = 1.0
            if i[1] == "DLFuzz":
                dlfuzz += 1
                dist_dlfuzz = 1.0
            if i[1] == "DeepHyperion":
                deephyperion += 1
                dist_deephyperion = 1.0

        list_data = [("DeepAtash", dist_deepatash , deepatash, target_deepatash),
            ("DLFuzz", dist_dlfuzz, dlfuzz, target_dlfuzz),
            ("DeepHyperion", dist_deephyperion, deephyperion, target_deephyperion)]

    return  list_data
  


def compare_with_dh_dlf(approach, div, features, target_area):

    if div == "HEATMAP":
        threshold = 0.09
    elif div == "INPUT":
        threshold = 4.8
    elif div == "LATENT":
        threshold = 0.01

    result_folder = f"../experiments/data/mnist/{target_area}"

    
    for feature in features:

        dst = f"../experiments/data/mnist/DeepAtash/{target_area}/{feature[0]}"
        dst_dh = f"../experiments/data/mnist/DeepHyperion/{feature[0]}"
        dst_dlf = f"../experiments/data/mnist/DLFuzz"
        
        for i in range(1, 11):
            # load approach data
            inputs_focused = load_data(dst, str(i)+"-", approach, div)
            print(len(inputs_focused))
            

            # load DH data
            for subdir, dirs, files in os.walk(dst_dh, followlinks=False):
                if str(i)+"-" in subdir and "all" in subdir:
                    inputs_dh = compute_targets_for_dh(subdir, feature[1], feature[0], threshold, div)
            
            # load DLFuzz data
            for subdir, dirs, files in os.walk(dst_dlf, followlinks=False):
                if str(i)+"-" in subdir:
                    inputs_dlf = compute_targets_for_dlf(dst_dlf, feature[1], feature[0], threshold, div)


            list_data = compute_tSNE_cluster_vs_dh_and_dlf(inputs_dh+inputs_dlf+inputs_focused, result_folder, feature[0], approach, div, str(i))


            for data in list_data:
                dict_data = {
                    "approach": data[0],
                    "run": i,
                    "test input count": data[2],
                    "features": feature[0],
                    "num tsne clusters": str(data[1]),
                    "test input count in target": data[3]
                }


                filedest = f"{result_folder}/report_{data[0]}_{feature[0]}_{target_area}_{i}.json"
                with open(filedest, 'w') as f:
                    (json.dump(dict_data, f, sort_keys=True, indent=4))
           


if __name__ == "__main__": 

    feature_combinations = ["Moves-Bitmaps", "Orientation-Bitmaps", "Orientation-Moves"] # []
    dst = f"../experiments/data/mnist/DeepAtash"
    find_best_div_approach(dst, feature_combinations)


    features = [("Moves-Bitmaps", (10/move_range, 140/bitmaps_range)), ("Orientation-Bitmaps",(160/orientation_range, 60/bitmaps_range)), ("Orientation-Moves",(0/move_range, 160/orientation_range))]
    compare_with_dh_dlf("nsga2", "LATENT", features, "target_cell_in_dark")


    features = [("Moves-Bitmaps", (10/move_range, 124/bitmaps_range)), ("Orientation-Bitmaps",(-30/orientation_range, 80/bitmaps_range)), ("Orientation-Moves",(12/move_range, 90/orientation_range))]
    compare_with_dh_dlf("nsga2", "LATENT", features, "target_cell_in_gray")


    features = [("Moves-Bitmaps", (13/move_range, 174/bitmaps_range)), ("Orientation-Bitmaps",(-30/orientation_range, 22/bitmaps_range)), ("Orientation-Moves",(3/move_range, -130/orientation_range))]
    compare_with_dh_dlf("nsga2", "LATENT", features, "target_cell_in_white")
