import os
import numpy as np
import tensorflow as tf
import json
import matplotlib
matplotlib.use('Agg')
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy

# local
from config import EXPECTED_LABEL
import vectorization_tools
from utils import move_distance, bitmap_count, orientation_calc
from feature import Feature
from sample import Sample
import rasterization_tools

BITMAP_THRESHOLD = 0.5
mnist = tf.keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()   

def _basic_feature_stats():
    return {
        'min' : np.PINF,
        'max' : np.NINF,
        'missing' : 0
    }

class FeatureMap:
    def __init__(self):
        self.samples = None
        self.features = None

    def extract_stats(self, images, features):

        # Iteratively walk in the dataset and process all the json files. For each of them compute the statistics
        data = {}
        # Overall Count
        data['total'] = 0
        # Features
        data['features'] = {f: _basic_feature_stats() for f in features}

        samples = []
        for img in images:   
            seed = img[0]         
            seed_image = img[1]
            xml_desc = vectorization_tools.vectorize(seed_image)
            sample = Sample(xml_desc, EXPECTED_LABEL, seed)
            # we need original image to compute prediction
            orig_image = rasterization_tools.rasterize_in_memory(vectorization_tools.vectorize(x_test[seed]))
            performance = sample.evaluate(orig_image)
            if performance < 0:
                misbehaviour = True
            else: 
                misbehaviour = False

            predicted_label = sample.predicted_label

            sample_dict = {
                "expected_label": str(EXPECTED_LABEL),
                "features": {
                    "moves":  move_distance(sample),
                    "orientation": orientation_calc(sample,0),
                    "bitmaps": bitmap_count(sample, BITMAP_THRESHOLD)
                },
                "id": sample.id,
                "misbehaviour": misbehaviour,
                "performance": str(performance),
                "predicted_label": predicted_label,
                "seed": seed 
            }

            sample.from_dict(sample_dict)
            samples.append(sample)
            print(".", end='', flush=True)

            # Total count
            data['total'] += 1

            # Process only the features that are in the sample
            for k, v in data['features'].items():
                # TODO There must be a pythonic way of doing it
                if k not in sample_dict["features"].keys():
                    v['missing'] += 1

                    # if report_missing_features:
                    print("Sample %s miss feature %s", sample_dict["id"], k)

                    continue

                if sample_dict["features"][k] != "None":
                    v['min'] = min(v['min'], sample_dict["features"][k])
                    v['max'] = max(v['max'], sample_dict["features"][k])

        for feature_name, feature_extrema in data['features'].items():
            parsable_string_tokens = ["=".join(["name",feature_name])]
        for extremum_name, extremum_value in feature_extrema.items():
            parsable_string_tokens.append("=".join([extremum_name, str(extremum_value)]))
        print(",".join(parsable_string_tokens))

        filedest = "MNIST.meta"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

        self.samples = samples
        return data

    def compute_featuremap_3d(self, features):  
        
        # Generate the map axes
        map_features = []
        for f in features:
            print("Using feature %s" % f[0])
            map_features.append(Feature(f[0], f[1], f[2], f[3], f[4]))

        feature1 = map_features[0]
        feature2 = map_features[1]
        feature3 = map_features[2]

        # Reshape the data as ndimensional array. But account for the lower and upper bins.
        archive_data = np.full([feature1.num_cells, feature2.num_cells, feature3.num_cells], None, dtype=object)
        # counts the number of samples in each cell
        coverage_data = np.zeros(shape=(10,feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

        misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

        for sample in fm.samples:
            # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
            x_coord = feature1.get_coordinate_for(sample) - 1
            y_coord = feature2.get_coordinate_for(sample) - 1
            z_coord = feature3.get_coordinate_for(sample) - 1

            if archive_data[x_coord, y_coord, z_coord] is None:
                arx = [sample]
                archive_data[x_coord, y_coord, z_coord] = arx
            else:
                archive_data[x_coord, y_coord, z_coord].append(sample)
            # Increment the coverage 
            coverage_data[int(sample.expected_label),x_coord, y_coord, z_coord] += 1

            if sample.is_misbehavior():
                # Increment the misbehaviour 
                misbehaviour_data[x_coord, y_coord, z_coord] += 1

        return archive_data, coverage_data, misbehaviour_data

    def compute_featuremap(self, features):      
        # Generate the map axes
        map_features = []
        for f in features:
            map_features.append(Feature(f[0], f[1], f[2], f[3], f[4]))

        feature1 = map_features[0]
        feature2 = map_features[1]

        # Reshape the data as ndimensional array. But account for the lower and upper bins.
        archive_data = np.full([feature1.num_cells, feature2.num_cells], None, dtype=object)
        # counts the number of samples in each cell
        coverage_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)

        misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)

        for sample in self.samples:
            x_coord = feature1.get_coordinate_for(sample) - 1
            y_coord = feature2.get_coordinate_for(sample) - 1

            if archive_data[x_coord, y_coord] is None:
                arx = [sample]
                archive_data[x_coord, y_coord] = arx
            else:
                archive_data[x_coord, y_coord].append(sample)
            # Increment the coverage 
            coverage_data[x_coord, y_coord] += 1

            if sample.is_misbehavior():
                # Increment the misbehaviour 
                misbehaviour_data[x_coord, y_coord] += 1

        return archive_data, coverage_data, misbehaviour_data
        
    def visualize(self, features, filename):
        """
            Visualize the samples and the features on a map. The map cells contains the number of samples for each
            cell, so empty cells (0) are white, cells with few elements have a light color, while cells with more
            elements have darker color. This gives an intuition on the distribution of the misbheaviors and the
            collisions
        Returns:
        """

        figures = []

        # Create one visualization for each pair of self.axes selected in order
        for feature1, feature2 in itertools.combinations(features, 2):

            _features = [feature1, feature2]
            _, coverage_data, misbehaviour_data = self.compute_featuremap(_features)
            
            feature1 = Feature(feature1[0], feature1[1], feature1[2], feature1[3], feature1[4])
            feature2 = Feature(feature2[0], feature2[1], feature2[2], feature2[3], feature2[4])

            # figure
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = sns.cubehelix_palette(dark=0.5, light=0.9, as_cmap=True)
            # Set the color for the under the limit to be white (so they are not visualized)
            cmap.set_under('1.0')

            # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
            # second on the x. So we transpose
            coverage_data = np.transpose(coverage_data)

            sns.heatmap(coverage_data, vmin=1, vmax=20, square=True, cmap=cmap)

            # Plot misbehaviors - Iterate over all the elements of the array to get their coordinates:
            it = np.nditer(misbehaviour_data, flags=['multi_index'])
            for v in it:
                # Plot only misbehaviors
                if v > 0:
                    alpha = 0.1 * v if v <= 10 else 1.0
                    (x, y) = it.multi_index
                    # Plot as scattered plot. the +0.5 ensures that the marker in centered in the cell
                    plt.scatter(x + 0.5, y + 0.5, color="black", alpha=alpha, s=50)

            xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels()]
            ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels()]
            #
            ax.set_xticklabels(xtickslabel)
            plt.xticks(rotation=45)
            ax.set_yticklabels(ytickslabel)
            plt.yticks(rotation=0)

            title_tokens = [f"{filename}: Collision Map Digit:", str(EXPECTED_LABEL), "\n"]

            the_title = " ".join(title_tokens)

            fig.suptitle(the_title, fontsize=16)

            # Plot small values of y below.
            # We need this to have the y axis start from zero at the bottom
            ax.invert_yaxis()

            # axis labels
            plt.xlabel(feature1.feature_name)
            plt.ylabel(feature2.feature_name)

            # Add the store_to attribute to the figure object
            store_to = "-".join([filename, str(EXPECTED_LABEL), feature1.feature_name, feature2.feature_name])
            setattr(fig, "store_to", store_to)

            figures.append(fig)

        file_format = 'pdf'
        for figure in figures:
            file_name_tokens = [figure.store_to]

            # Add File extension
            figure_file_name = "-".join(file_name_tokens) + "." + file_format

            figure_file = os.path.join(f"logs/{EXPECTED_LABEL}", figure_file_name)

            figure.savefig(figure_file, format=file_format)

    def visualize_probability(self, features, filename):
        """
            Visualize the samples and the features on a map. The map cells contains the number of samples for each
            cell, so empty cells (0) are white, cells with few elements have a light color, while cells with more
            elements have darker color. This gives an intuition on the distribution of the misbheaviors and the
            collisions
        Returns:
        """

        figures = []

        # Create one visualization for each pair of self.axes selected in order
        for feature1, feature2 in itertools.combinations(features, 2):

            _features = [feature1, feature2]
            
            feature1 = Feature(feature1[0], feature1[1], feature1[2], feature1[3], feature1[4])
            feature2 = Feature(feature2[0], feature2[1], feature2[2], feature2[3], feature2[4])

            _, coverage_data, misbehaviour_data = self.compute_featuremap(_features)

            # figure
            fig, ax = plt.subplots(figsize=(8, 8))

            cmap = sns.cubehelix_palette(dark=0.1, light=0.9, as_cmap=True)
            # Cells have a value between 0.0 and 1.0 since they represent probabilities

            # Set the color for the under the limit to be white (0.0) so empty cells are not visualized
            # cmap.set_under('0.0')
            # Plot NaN in white
            cmap.set_bad(color='white')

            # Coverage data might be zero, so this produces Nan. We convert that to 0.0
            # probability_data = np.nan_to_num(misbehaviour_data / coverage_data)
            raw_probability_data = misbehaviour_data / coverage_data

            # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
            # second on the x. So we transpose
            probability_data = np.transpose(raw_probability_data)
            # find max probabilities
            prob = np.nan_to_num(probability_data)
            indices = np.where(prob == np.amax(prob))

            max_indices = list(zip(indices[0], indices[1]))

            sns.heatmap(probability_data, vmin=0.0, vmax=1.0, square=True, cmap=cmap)

            xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels()]
            ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels()]
            #
            ax.set_xticklabels(xtickslabel)
            plt.xticks(rotation=45)
            ax.set_yticklabels(ytickslabel)
            plt.yticks(rotation=0)

            title_tokens = [f"{filename} Misbehaviour Probability","Digit " + str(EXPECTED_LABEL), "\n"]

            the_title = " ".join(title_tokens)

            fig.suptitle(the_title, fontsize=16)

            # Plot small values of y below.
            # We need this to have the y axis start from zero at the bottom
            ax.invert_yaxis()

            # axis labels
            plt.xlabel(feature1.feature_name)
            plt.ylabel(feature2.feature_name)

            # Include data to store the file with same prefix

            # Add the store_to attribute to the figure and maps object
            setattr(fig, "store_to", "-".join([filename, str(EXPECTED_LABEL), "probability", feature1.feature_name, feature2.feature_name]))
            figures.append(fig)
            
            _entropy = entropy(np.nan_to_num(probability_data.flatten()))
            print(filename, feature1.feature_name, feature2.feature_name, _entropy)


        file_format = 'pdf'
        for figure in figures:
            file_name_tokens = [figure.store_to]

            # Add File extension
            figure_file_name = "-".join(file_name_tokens) + "." + file_format

            figure_file = os.path.join(f"logs/{EXPECTED_LABEL}", figure_file_name)

            figure.savefig(figure_file, format=file_format)  

        
        return max_indices 

