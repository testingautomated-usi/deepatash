from plotting_utils import create_custom_palette,  \
    filter_data_and_plot_as_boxplots, store_figure_to_paper_folder, rename_features, create_the_table_retrain

import matplotlib.pyplot as plt
import os
import json
import pandas as pd



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

                map_dict["Tool"] = map_dict["approach"]
                map_dict["accuracy test set"] = round(float(map_dict["accuracy test set"]), 4)
                map_dict["accuracy target test set"] = round(float(map_dict["accuracy target test set"]), 4)

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
    tools = ["Before", "After"] 
    fontsize = 20
    min_fontsize = 16
    color_palette = create_custom_palette()

    # Create the figure
    fig = plt.figure(figsize=(16, 10))
    # Set the figure to be a grid with 1 column and 2 rows without space between them
    gs = fig.add_gridspec(2, hspace=0)
    # Get the axes objects
    axs = gs.subplots(sharex=True)
    # Plot the top plot
    axs[0] = filter_data_and_plot_as_boxplots(axs[0], "accuracy test set", plot_data, color_palette, tools)
    # Plot the bottom plot
    axs[1] = filter_data_and_plot_as_boxplots(axs[1], "accuracy target test set", plot_data, color_palette, tools)
    # Plot the bottom plot


    # Adjust the plots

    # Put a legend to the right of the current axis
    axs[0].legend(bbox_to_anchor=(1.04,1), loc="upper left",borderaxespad=0,fontsize=fontsize)
    # Increase font for y-label and y-ticks
    # axs[0].set_ylabel(axs[0].get_ylabel(), fontsize=fontsize)
    axs[0].set_ylabel("ACC test set", fontsize=fontsize)
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
    axs[1].set_ylabel("ACC target set", fontsize=fontsize)
    axs[1].tick_params(axis='y', which='major', labelsize=min_fontsize)

    # Align the y labels: -0.1 moves it a bit to the left, 0.5 move it in the middle of y-axis
    axs[0].get_yaxis().set_label_coords(-0.08, 0.5)
    axs[1].get_yaxis().set_label_coords(-0.08, 0.5)

   
    # Align the y labels: -0.1 moves it a bit to the left, 0.5 move it in the middle of y-axis
    axs[0].get_yaxis().set_label_coords(-0.08, 0.5)
    axs[1].get_yaxis().set_label_coords(-0.08, 0.5)


    return fig




if __name__ == "__main__": 
    mnist_data = load_data_from_folder("./data/mnist/retrain")
    mnist_figure = preprare_the_figure(mnist_data)

    # Store
    create_the_table_retrain(mnist_data, file_name=f"RQ3-MNIST")
    store_figure_to_paper_folder(mnist_figure, file_name=f"RQ3-MNIST")


    imdb_data = load_data_from_folder("./data/imdb/retrain")
    imdb_figure = preprare_the_figure(imdb_data)

    # Store
    create_the_table_retrain(imdb_data, file_name=f"RQ3-IMDB")
    store_figure_to_paper_folder(imdb_figure, file_name=f"RQ3-IMDB")





