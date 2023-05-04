import os
from config import META_FILE
import numpy as np
import json
import matplotlib.pyplot as plt

def compute_percentile(arr, percentile=0.25):
    arr.sort()
    percentile_index = int(len(arr) * percentile)
    return arr[percentile_index]


def compute_candidate_indices(arr, threshold):
    """Returns indices from a numpy array smaller than median"""
    indices = np.flatnonzero(arr < threshold)
    return np.unravel_index(indices, arr.shape)

def extract_candidate_targets(dst, feature):
    coverages = []
    probabilities = []
    
    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for npy_path in [os.path.join(subdir, f) for f in files if f.startswith("coverage") and f.endswith(".npy")]:
            if feature in npy_path:
                coverage = np.load(npy_path)
                coverages.append(coverage)

    for subdir, dirs, files in os.walk(dst, followlinks=False):
        # Consider only the files that match the pattern
        for npy_path in [os.path.join(subdir, f) for f in files if f.startswith("probability") and f.endswith(".npy")]:
            if feature in npy_path:
                probability = np.load(npy_path)
                probabilities.append(probability)
    

    average_coverage = np.nanmean(np.array(coverages), axis=0)

    average_probability = np.nanmean(np.array(probabilities), axis=0)

    average_probability = np.nan_to_num(average_probability)

    mean_coverage = np.nanmean(average_coverage)

    percentile_25 = compute_percentile(average_coverage[average_coverage > 0].flatten(), 0.25)

    candidate_indices = compute_candidate_indices(average_coverage, mean_coverage)


    candidate_targets_dark = []
    candidate_targets_grey = []
    candidate_targets_white = []


    for i in range(0,len(candidate_indices[0])):
        x = candidate_indices[0][i]
        y = candidate_indices[1][i]
        if average_coverage[x,y] == 0:
            candidate_targets_white.append([(x, y), average_coverage[x,y]]) 
        elif average_probability[x,y] > 0.8:
            candidate_targets_dark.append([(x, y), average_coverage[x,y]])
        elif average_probability[x,y] > 0:
            candidate_targets_grey.append([(x, y), average_coverage[x,y]]) 


    return candidate_targets_dark, candidate_targets_grey, candidate_targets_white, mean_coverage, percentile_25


if __name__ == "__main__": 
    # you need to compute coverage maps of DeepHyperion and stats before hand
    dst = "../experiments/data/mnist/DeepHyperion"
    features = ["moves-bitmaps","orientation-bitmaps", "moves-orientation"]
    
    for feature in features:
        candidate_targets_dark, candidate_targets_grey, candidate_targets_white, median_coverage, percentile_25 = extract_candidate_targets(dst, feature)
        print(f"features: {feature}")
        print(f"mean coverage: {median_coverage}")
        print(f"percentile 25: {percentile_25}")
        print(f"candidate targets dark: ")
        print(candidate_targets_dark)
        print(f"candidate targets grey: ")
        print(candidate_targets_grey)
        print(f"candidate targets white: ")
        print(candidate_targets_white)


