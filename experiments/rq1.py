from plotting_utils import create_custom_palette, create_the_table, \
    filter_data_and_plot_as_boxplots, store_figure_to_paper_folder, rename_features, create_the_table

import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
import csv


def load_data_from_folder(dataset_folder, allowed_features_combination=None):
    """
    Returns: Panda DF with the data about the experiments from the data folder, data/mnist or data/beamng. Merge the configurations of DH together
    -------
    """

    the_data = None

    for subdir, dirs, files in os.walk(dataset_folder, followlinks=False):

        # Consider only the files that match the pattern
        for json_data_file in [os.path.join(subdir, f) for f in files if f.startswith("report") and f.endswith(".json")]: 

            with open(json_data_file, 'r') as input_file:
                # Get the JSON
                map_dict = json.load(input_file)

                target_feature_combination = "-".join(rename_features(map_dict["features"].replace("_", "-").split("-")))

                map_dict["Features Combination"] = target_feature_combination

                map_dict["Tool"] = map_dict["approach"]+"-"+map_dict["diversity"]

                map_dict["num tsne clusters"] = float(map_dict["num tsne clusters"])

                if "target num tsne clusters" in map_dict:
                    map_dict["target num tsne clusters"] = float(map_dict["target num tsne clusters"])
                if "target num tsne clustes" in map_dict:
                    map_dict["target num tsne clusters"] = float(map_dict["target num tsne clusters"])

                if the_data is None:
                    # Creates the DataFrame
                    the_data = pd.json_normalize(map_dict)
                else:
                    # Maybe better to concatenate only once
                    the_data = pd.concat([the_data, pd.json_normalize(map_dict)])


    # Fix data type
    print("Loaded data for:", the_data["approach"].unique())
    print("\tFeatures Combinations:", the_data["Features Combination"].unique())
    return the_data

def preprare_the_figure(plot_data):
    tools = [ "GA-Input", "NSGA2-Input", "GA-Latent", "NSGA2-Latent", "GA-Heatmap", "NSGA2-Heatmap"]
    fontsize = 20
    min_fontsize = 16
    color_palette = create_custom_palette()

    # Create the figure
    fig = plt.figure(figsize=(16, 10))
    # Set the figure to be a grid with 1 column and 4 rows without space between them
    gs = fig.add_gridspec(4, 1, hspace=0)
    # Get the axes objects
    axs = gs.subplots(sharex=True)
    # Plot the top plot
    axs[0] = filter_data_and_plot_as_boxplots(axs[0], "test input count", plot_data, color_palette, tools)
    # Plot the bottom plot
    axs[1] = filter_data_and_plot_as_boxplots(axs[1], "num tsne clusters", plot_data, color_palette, tools)
    # Plot the bottom plot
    axs[2] = filter_data_and_plot_as_boxplots(axs[2], "test input count in target", plot_data, color_palette, tools)
    # Plot the bottom plot
    axs[2] = filter_data_and_plot_as_boxplots(axs[3], "target num tsne clusters", plot_data, color_palette, tools)

    # Adjust the plots

    # Put a legend to the right of the current axis
    axs[0].legend(bbox_to_anchor=(1.04,1), loc="upper left",borderaxespad=0,fontsize=fontsize)
    # Increase font for y-label and y-ticks
    # axs[0].set_ylabel(axs[0].get_ylabel(), fontsize=fontsize)
    axs[0].set_ylabel("Test Count", fontsize=fontsize)
    axs[0].tick_params(axis='y', which='major', labelsize=min_fontsize)
    # ax.tick_params(axis='both', which='minor', labelsize=8)
    #axs[0].legend(fontsize=fontsize)

    # Remove the legend from the bottom plot
    axs[1].legend([], [], frameon=False)
    # Remove the x - label
    axs[1].set_xlabel('')
    # Increase only the size of x-ticks, but split the combinations in two lines
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # https://stackoverflow.com/questions/11244514/modify-tick-label-text
    axs[1].set_xticklabels([l.get_text().replace("-", "\n") for l in axs[1].get_xticklabels()], fontsize=fontsize)
    # Increase label y-label and y-ticks
    # axs[1].set_ylabel(axs[1].get_ylabel(), fontsize=fontsize)
    axs[1].set_ylabel("Test Sparseness", fontsize=fontsize)
    axs[1].tick_params(axis='y', which='major', labelsize=min_fontsize)


    # Remove the legend from the bottom plot
    axs[2].legend([], [], frameon=False)
    # Remove the x - label
    axs[2].set_xlabel('')
    # Increase only the size of x-ticks, but split the combinations in two lines
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # https://stackoverflow.com/questions/11244514/modify-tick-label-text
    axs[2].set_xticklabels([l.get_text().replace("-", "\n") for l in axs[2].get_xticklabels()], fontsize=fontsize)
    # Increase label y-label and y-ticks
    # axs[1].set_ylabel(axs[1].get_ylabel(), fontsize=fontsize)
    axs[2].set_ylabel("Test Target", fontsize=fontsize)
    axs[2].tick_params(axis='y', which='major', labelsize=min_fontsize)

    # Remove the legend from the bottom plot
    axs[3].legend([], [], frameon=False)
    # Remove the x - label
    axs[3].set_xlabel('')
    # Increase only the size of x-ticks, but split the combinations in two lines
    # labels = [item.get_text() for item in ax.get_xticklabels()]
    # labels[1] = 'Testing'
    # https://stackoverflow.com/questions/11244514/modify-tick-label-text
    axs[3].set_xticklabels([l.get_text().replace("-", "\n") for l in axs[2].get_xticklabels()], fontsize=fontsize)
    # Increase label y-label and y-ticks
    # axs[1].set_ylabel(axs[1].get_ylabel(), fontsize=fontsize)
    axs[3].set_ylabel("Target Sparseness", fontsize=fontsize)
    axs[3].tick_params(axis='y', which='major', labelsize=min_fontsize)

    # Align the y labels: -0.1 moves it a bit to the left, 0.5 move it in the middle of y-axis
    axs[0].get_yaxis().set_label_coords(-0.08, 0.5)
    axs[1].get_yaxis().set_label_coords(-0.08, 0.5)
    axs[2].get_yaxis().set_label_coords(-0.08, 0.5)
    axs[3].get_yaxis().set_label_coords(-0.08, 0.5)

    return fig


if __name__ == "__main__": 

    # mnist_data = load_data_from_folder("./data/mnist/DeepAtash")
    # mnist_figure = preprare_the_figure(mnist_data)

    # # Store
    # create_the_table(mnist_data, file_name=f"RQ1-MNIST-all")
    # store_figure_to_paper_folder(mnist_figure, file_name=f"RQ1-MNIST-all")

    # mnist_data = load_data_from_folder("./data/mnist/DeepAtash/target_cell_in_dark")
    # mnist_figure = preprare_the_figure(mnist_data)

    # # Store
    # create_the_table(mnist_data, file_name=f"RQ1-MNIST-dark-table")
    # store_figure_to_paper_folder(mnist_figure, file_name=f"RQ1-MNIST-dark")


    # mnist_data = load_data_from_folder("./data/mnist/DeepAtash/target_cell_in_grey")
    # mnist_figure = preprare_the_figure(mnist_data)

    # # Store
    # create_the_table(mnist_data, file_name="RQ1-MNIST-grey-table")
    # store_figure_to_paper_folder(mnist_figure, file_name="RQ1-MNIST-grey")

    # mnist_data = load_data_from_folder("./data/mnist/DeepAtash/target_cell_in_white")
    # mnist_figure = preprare_the_figure(mnist_data)

    # # Store
    # create_the_table(mnist_data, file_name=f"RQ1-MNIST-white-table")
    # store_figure_to_paper_folder(mnist_figure, file_name=f"RQ1-MNIST-white")

    # imdb_data = load_data_from_folder("./data/imdb/DeepAtash")
    # imdb_figure = preprare_the_figure(imdb_data)

    # # Store
    # create_the_table(imdb_data, file_name=f"RQ1-IMDB-all-table")
    # store_figure_to_paper_folder(imdb_figure, file_name=f"RQ1-IMDB-all")


    # imdb_data = load_data_from_folder("./data/imdb/DeepAtash/target_cell_in_dark")
    # imdb_figure = preprare_the_figure(imdb_data)

    # # Store
    # create_the_table(imdb_data, file_name=f"RQ1-IMDB-dark-table")
    # store_figure_to_paper_folder(imdb_figure, file_name=f"RQ1-IMDB-dark")


    # imdb_data = load_data_from_folder("./data/imdb/DeepAtash/target_cell_in_grey")
    # imdb_figure = preprare_the_figure(imdb_data)

    # # Store
    # create_the_table(imdb_data, file_name="RQ1-IMDB-grey-table")
    # store_figure_to_paper_folder(imdb_figure, file_name="RQ1-IMDB-grey")

    # imdb_data = load_data_from_folder("./data/imdb/DeepAtash/target_cell_in_white")
    # imdb_figure = preprare_the_figure(imdb_data)

    # # Store
    # create_the_table(imdb_data, file_name=f"RQ1-IMDB-white-table")
    # store_figure_to_paper_folder(imdb_figure, file_name=f"RQ1-IMDB-white")


    # beamng_data = load_data_from_folder("./data/bng/DeepAtash/target_cell_in_dark")
    # beamng_figure = preprare_the_figure(beamng_data)

    # # Store
    # create_the_table(beamng_data, file_name="RQ1-BeamNG-dark-table")
    # store_figure_to_paper_folder(beamng_figure, file_name="RQ1-BeamNG-dark")

    beamng_data = load_data_from_folder("./data/bng/DeepAtash/target_cell_in_grey")
    beamng_figure = preprare_the_figure(beamng_data)

    # Store
    create_the_table(beamng_data, file_name="RQ1-BeamNG-grey-table")
    store_figure_to_paper_folder(beamng_figure, file_name="RQ1-BeamNG-grey")


    # beamng_data = load_data_from_folder("./data/bng/DeepAtash/target_cell_in_white")
    # beamng_figure = preprare_the_figure(beamng_data)

    # # Store
    # create_the_table(beamng_data, file_name="RQ1-BeamNG-white-table")
    # store_figure_to_paper_folder(beamng_figure, file_name="RQ1-BeamNG-white")