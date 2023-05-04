import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import json
import pickle
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Tuple, List
from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib
from pathlib import Path
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

# local
from evaluator import Evaluator
import utils as us
from sample import Sample
from config import INPUT_MAXLEN, EXPECTED_LABEL, MODEL, META_FILE, TARGET_SIZE
from feature import Feature

model = tf.keras.models.load_model(MODEL)
doc2vec = Doc2Vec.load("models/Doc2Vec_dbow,d100,n5,mc2,t16")
evaluator = Evaluator()

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

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


def load_data_all(dst, features):
    """
    :param dst: source folder for DeepAtash experiments
    :param features: feature combinations to consider
    :return: list of misbehaviours in the archives of DeepAtash with different configurations
    """
    inputs = []
    inputs_target = []
    for subdir, _, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
            for json_path in [os.path.join(subdir, f) for f in files if f.startswith("mbr") and f.endswith(".json")]:
                if features in json_path:  
                    print(".", end='', flush=True)
  

                    if "ga_" in json_path:
                        y2 = "GA"
                    elif "nsga2" in json_path:
                        y2 = "NSGA2"

                    if "LATENT" in json_path:
                        y1 = "LATENT"
                    if "INPUT" in json_path:
                        y1 = "INPUT"
                    if "HEATMAP" in json_path:
                        y1 = "HEATMAP" 
    
                                         
                    with open(json_path) as jf:
                        json_data = json.load(jf)

                    if json_data["misbehaviour"] == True:
                        seq = tokenizer.texts_to_sequences([json_data["text"]])
                        padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
                        inputs.append([padded_texts[0], f"{y2}-{y1}", json_data['predicted_label'], float(json_data["distance to target"]), json_data["elapsed"]])
                        if float(json_data["distance to target"]) == 0:
                            inputs_target.append([padded_texts[0], f"{y2}-{y1}", json_data['predicted_label'], float(json_data["distance to target"]), json_data["elapsed"]])

   
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
    for subdir, _, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        if i+approach in subdir and div in subdir:
            for json_path in [os.path.join(subdir, f) for f in files if f.startswith("mbr") and f.endswith(".json")]:
                    print(".", end='', flush=True)  

                    if "ga_" in json_path:
                        y2 = "GA"
                    elif "nsga2" in json_path:
                        y2 = "NSGA2"

                    if "LATENT" in json_path:
                        y1 = "LATENT"
                    if "INPUT" in json_path:
                        y1 = "INPUT"
                    if "HEATMAP" in json_path:
                        y1 = "HEATMAP" 
    
                                         
                    with open(json_path) as jf:
                        json_data = json.load(jf)

                    if json_data["misbehaviour"] == True:
                        seq = tokenizer.texts_to_sequences([json_data["text"]])
                        padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
                        inputs.append([padded_texts[0], f"{y2}-{y1}", json_data['predicted_label'], float(json_data["distance to target"]), json_data["elapsed"]])
                        if float(json_data["distance to target"]) == 0:
                            inputs_target.append([padded_texts[0], f"{y2}-{y1}", json_data['predicted_label'], float(json_data["distance to target"]), json_data["elapsed"]])

   
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
    X, y, txts = [], [], []
    for i in inputs:
        X.append(i[0].flatten())
        y.append(i[1])
        txts.append(i[2])
    
    X = np.array(X)

    feat_cols = ['word'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X,columns=feat_cols)
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
        data=df,
        legend="full",
        alpha=0.3
    )
    fig.savefig(f"{_folder}/tsne-diag-{features}-{div}-{ii}-1.pdf", format='pdf')

    return df


def compute_tSNE_and_cluster_all(inputs, targets, _folder, features, div, ii=0):

    target_input_ga = 0
    target_input_nsga2 = 0
    target_latent_ga = 0
    target_latent_nsga2 = 0
    target_heatmap_ga = 0
    target_heatmap_nsga2 = 0

    div_input_ga = 0
    div_input_nsga2 = 0
    div_latent_ga = 0
    div_latent_nsga2 = 0
    div_heatmap_ga = 0
    div_heatmap_nsga2 = 0

    div_target_input_ga = 0
    div_target_input_nsga2 = 0
    div_target_latent_ga = 0
    div_target_latent_nsga2 = 0
    div_target_heatmap_ga = 0
    div_target_heatmap_nsga2 = 0

    input_ga = 0
    input_nsga2 = 0
    latent_ga = 0
    latent_nsga2 = 0
    heatmap_ga = 0
    heatmap_nsga2 = 0

    for i in targets:
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


        np_data_cols = df.iloc[:,2003:2005]

        n = len(inputs) - 1

        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-1.pdf", format='pdf')

        
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
        div_heatmap_nsga2 = df_nsga2_heatmap.nunique()['Clusters']/num_clusters


        input_ga = len(df_ga_input.index)
        input_nsga2 = len(df_nsga2_input.index)

        latent_ga = len(df_ga_latent.index)
        latent_nsga2 = len(df_nsga2_latent.index)

        heatmap_ga = len(df_ga_heatmap.index)
        heatmap_nsga2 = len(df_nsga2_heatmap.index)        

    else:
        for i in inputs:
            if i[1] == "GA-INPUT":
                input_ga += 1
                div_input_ga = 1.0
            if i[1] == "NSGA2-INPUT":
                input_nsga2 += 1
                div_input_nsga2 = 1.0
            if i[1] == "GA-LATENT":
                latent_ga += 1
                div_latent_ga = 1.0
            if i[1] == "NSGA2-LATENT":
                latent_nsga2 += 1
                div_latent_nsga2 = 1.0
            if i[1] == "GA-HEATMAP":
                heatmap_ga += 1
                div_heatmap_ga = 1.0
            if i[1] == "NSGA2-HEATMAP":
                heatmap_nsga2 += 1 
                div_heatmap_nsga2 = 1.0   

    if len(targets) > 3:

        df_target = plot_tSNE(targets, _folder, features, div, ii)
        df_target = df_target.reset_index()  # make sure indexes pair with number of rows

        np_data_cols = df_target.iloc[:,2003:2005]

        n = len(targets) - 1

        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df_target
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-tt-{features}-{div}-{ii}-1.pdf", format='pdf')

        
        df_target_nsga2_input = df_target[df_target.label == "NSGA2-INPUT"]
        df_target_ga_input = df_target[df_target.label =="GA-INPUT"]

        df_target_nsga2_latent = df_target[df_target.label == "NSGA2-LATENT"]
        df_target_ga_latent = df_target[df_target.label =="GA-LATENT"]

        df_target_nsga2_heatmap = df_target[df_target.label == "NSGA2-HEATMAP"]
        df_target_ga_heatmap = df_target[df_target.label =="GA-HEATMAP"]


        num_clusters2 = len(centers)

        div_target_input_ga = df_target_ga_input.nunique()['Clusters']/num_clusters2
        div_target_input_nsga2 = df_target_nsga2_input.nunique()['Clusters']/num_clusters2

        div_target_latent_ga = df_target_ga_latent.nunique()['Clusters']/num_clusters2
        div_target_latent_nsga2 = df_target_nsga2_latent.nunique()['Clusters']/num_clusters2

        div_target_heatmap_ga = df_target_ga_heatmap.nunique()['Clusters']/num_clusters2
        div_target_heatmap_nsga2 = df_target_nsga2_heatmap.nunique()['Clusters']/num_clusters2
    else:
        for i in targets:
            if i[1] == "GA-INPUT":
                div_target_input_ga = 1.0
            if i[1] == "NSGA2-INPUT":
                div_target_input_nsga2 = 1.0
            if i[1] == "GA-LATENT":
                div_target_latent_ga = 1.0
            if i[1] == "NSGA2-LATENT":
                div_target_latent_nsga2 = 1.0
            if i[1] == "GA-HEATMAP":
                div_target_heatmap_ga = 1.0
            if i[1] == "NSGA2-HEATMAP":
                div_target_heatmap_nsga2 = 1.0

    list_data = [("GA", "Input", div_input_ga,  input_ga, target_input_ga, div_target_input_ga), ("NSGA2", "Input", div_input_nsga2, input_nsga2, target_input_nsga2, div_target_input_nsga2), \
                        ("GA", "Latent", div_latent_ga,  latent_ga, target_latent_ga, div_target_latent_ga), ("NSGA2", "Latent", div_latent_nsga2,  latent_nsga2, target_latent_nsga2, div_target_latent_nsga2), \
                        ("GA", "Heatmap",  div_heatmap_ga,  heatmap_ga, target_heatmap_ga, div_target_heatmap_ga), ("NSGA2", "Heatmap", div_heatmap_nsga2, heatmap_nsga2, target_heatmap_nsga2, div_target_heatmap_nsga2)]
    return list_data


def find_best_div_approach(dst, feature_combinations):

    evaluation_area = ["target_cell_in_dark", "target_cell_in_grey", "target_cell_in_white"] 

    for evaluate in evaluation_area:        
        for features in feature_combinations:
            for i in range(1, 11):
                inputs = []
                targets = []
                for subdir, _, _ in os.walk(dst, followlinks=False):
                    if features in subdir and str(i)+"-" in subdir and evaluate in subdir:
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
        
    if "NegCount" in FEATURES:
        f3 = Feature("negcount", meta["negcount"]["min"], meta["negcount"]["max"], "count_neg", 25)
        features.append(f3)
    if "PosCount" in FEATURES:
        f2 = Feature("poscount",meta["poscount"]["min"], meta["poscount"]["max"], "count_pos", 25)
        features.append(f2)
    if "VerbCount" in FEATURES:
        f1 = Feature("verbcount",meta["verbcount"]["min"], meta["verbcount"]["max"], "count_verbs", 25)
        features.append(f1)
    return features
 

def compute_targets_for_dh(dst, goal, features, metric):
    fts = generate_features(features)
    count = 0
    samples = []
    archive_samples = []
    for subdir, _, files in os.walk(dst, followlinks=False):
        for json_path in [os.path.join(subdir, f) for f in files if
                        (
                            f.startswith("mbr") and
                            f.endswith(".json")
                        )]:
            with open(json_path) as jf:
                ind = json.load(jf)
        

            sample = Sample(ind["text"], EXPECTED_LABEL, int(ind["seed"]))
            sample.predicted_label = ind["predicted_label"]
            sample.ff = float(ind["performance"])
            sample.elapsed = ind["elapsed"]


            sample.features["negcount"] = ind["features"]["negcount"]
            sample.features["poscount"] =  ind["features"]["poscount"]
            sample.features["verbcount"] = ind["features"]["verbcount"]

    
            b = tuple()
            for ft in fts:
                i = ft.get_coordinate_for(sample)
                if i != None:
                    b = b + (i,)
                else:
                    b = np.inf
            
            sample.distance_to_target = us.manhattan(b, goal)
            if sample.distance_to_target <= 1:
                # sample.compute_explanation()
                sample.compute_latent_vector(doc2vec)
                print(".", end='', flush=True)
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
                    if sample.ff < c.ff:
                        archive.append(sample)
                        archive[idx_min].sparseness = dmin
                    # c and ind have the same performance
                    elif sample.ff == c.ff:
                        # ind has better sparseness                        
                        if dmin > c.sparseness:
                            archive.append(sample)
                            archive[idx_min].sparseness = dmin



    target_samples = []
    for sample in archive:
        if sample.is_misbehavior() == True:
            seq = tokenizer.texts_to_sequences([sample.text])
            padded_texts = pad_sequences(seq, maxlen=INPUT_MAXLEN)
            archive_samples.append([padded_texts[0], f"DeepHyperion", sample.predicted_label, sample.distance_to_target, sample.elapsed])
            if sample.distance_to_target == 0:
                target_samples.append([padded_texts[0], f"DeepHyperion", sample.predicted_label, sample.distance_to_target, sample.elapsed])
             
    print("DeepHyperion", features, len(target_samples))
    return archive_samples, target_samples


def elapsed_to_millisec(elapsed):
    # compute milli seconds for sample's elapsed time
    times = re.split(r"[:.]", elapsed)
    millisecs = float(times[0])*3600+float(times[1])*60+float(times[2])+float(times[3])/1000000.
    return millisecs


def compute_metrics(inputs_da, targets_da, inputs_dh, targets_dh, _folder, features, approach, div, ii=0):
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
    
    # compute cluster coverage for sparseness

    inputs = inputs_da + inputs_dh
    if len(inputs) > 3:

        df = plot_tSNE(inputs, _folder, features, div, ii)
        df = df.reset_index()  # make sure indexes pair with number of rows

        np_data_cols = df.iloc[:,2003:2005]

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

        df_target = plot_tSNE(targets, _folder, features, div, ii)
        df_target = df_target.reset_index()  # make sure indexes pair with number of rows

        np_data_cols = df_target.iloc[:,2003:2005]

        n = len(targets) - 1

        labels, centers = cluster_data(data=np_data_cols, n_clusters_interval=(2, n))

        data_with_clusters = df_target
        data_with_clusters['Clusters'] = np.array(labels)
        fig = plt.figure(figsize=(10, 10))
        plt.scatter(data_with_clusters['tsne-2d-one'],data_with_clusters['tsne-2d-two'], c=data_with_clusters['Clusters'], cmap='rainbow')
        fig.savefig(f"{_folder}/cluster-diag-{features}-{div}-{ii}-1.pdf", format='pdf')

        
        df_da = df_target[df_target.label == f"{approach}-{div}"]
        df_dh = df_target[df_target.label =="DeepHyperion"]

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

    return  list_data
  
  
def compare_with_dh(approach, div, features, target_area):


    result_folder = f"../experiments/data/imdb/comparison/{target_area}"
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    
    for feature in features:

        dst = f"../experiments/data/imdb/DeepAtash/{target_area}/{feature[0]}"
        dst_dh = f"../experiments/data/imdb/DeepHyperion/{feature[0]}"
        
        for i in range(1, 11):
            # load approach data
            inputs_focused, target_focused = load_data(dst, str(i)+"-", approach, div)
            print(len(inputs_focused))
            

            # load DH data
            for subdir, dirs, files in os.walk(dst_dh, followlinks=False):
                if str(i)+"-" in subdir and "all" in subdir:
                    inputs_dh, targets_dh = compute_targets_for_dh(subdir, feature[1], feature[0], div)
            
            if "ga" == approach:
                approach2 = "GA"
            elif "nsga2" == approach:
                approach2 = "NSGA2"

            list_data = compute_metrics(inputs_focused, target_focused, inputs_dh, targets_dh, result_folder, feature[0], approach2, div, str(i))


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
        features =  [("NegCount-PosCount", (11, 8)), ("PosCount-VerbCount", (1, 7)), ("NegCount-VerbCount", (4, 12))]
        compare_with_dh("nsga2", "LATENT", features, "target_cell_in_dark")
    elif sys.argv[1] == "grey":
        features =  [("NegCount-VerbCount", (4, 11)), ("NegCount-PosCount",(14, 6)), ("PosCount-VerbCount", (8, 12))]
        compare_with_dh("nsga2", "LATENT", features, "target_cell_in_grey")
    elif sys.argv[1] == "white":
        features = [("NegCount-PosCount",(11, 6)), ("PosCount-VerbCount", (11, 13)), ("NegCount-VerbCount", (11, 6))]
        compare_with_dh("nsga2", "LATENT", features, "target_cell_in_white")

    else:
        feature_combinations = ["NegCount-PosCount", "PosCount-VerbCount", "NegCount-VerbCount"] 
        dst = f"../experiments/data/imdb/DeepAtash"
        find_best_div_approach(dst, feature_combinations)