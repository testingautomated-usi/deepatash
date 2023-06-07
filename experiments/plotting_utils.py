import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import errno
import statsmodels.stats.proportion as smp
from matplotlib.patches import Rectangle
from data_analysis import _log_statistics


def create_custom_palette():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = ["mistyrose", "rosybrown", "lightcoral", "indianred", "firebrick", "red", "orangered", "coral"] #prop_cycle.by_key()['color']
    tool_colors = {
        "DeepHyperion": colors[0],
        "DeepAtash": colors[4],  # C0C0C0 - #DCDCDC

        "NSGA2-Input": colors[0],
        "GA-Input": colors[1],
        "NSGA2-Latent": colors[2],
        "GA-Latent": colors[3],
        "NSGA2-Heatmap": colors[4],
        "GA-Heatmap": colors[5],
        "Before": colors[6],
        "After": colors[7]
    }
    return tool_colors


def rename_features(features):
    return [rename_feature(f) for f in features]


def rename_feature(feature):
    if "Bitmaps" == feature or "bitmaps" == feature:
        return "Lum"
    elif "Moves" == feature or "moves" == feature:
        return "Mov"
    elif "Orientation" == feature or "orientation" == feature:
        return "Or"
    ##
    elif "SegmentCount" == feature or "segment_count" == feature:
        return "TurnCnt"
    elif "MinRadius" == feature or "min_radius" == feature:
        return "MinRad"
    elif "MeanLateralPosition" == feature or "mean_lateral_position" == feature:
        return "MLP"
    elif "SDSteeringAngle" == feature or "sd_steering" == feature:
        return "StdSA"
    elif "Curvature" == feature or "curvature" == feature:
        return "Curv"
    ##
    if "PosCount" == feature or "poscount" == feature:
        return "Pos"
    elif "NegCount" == feature or "negcount" == feature:
        return "Neg"
    elif "VerbCount" == feature or "verbcount" == feature:
        return "Verb"    


# Utility to plot maps data
def filter_data_and_plot_as_boxplots(use_ax, we_plot, raw_data, palette, tools):

    assert type(we_plot) is str, "we_plot not a string !"

    # Select only the data we need to plot
    plot_axis_and_grouping = [
        "Tool",  # Test Subjects
        "Features Combination"  # Features that define this map
    ]
    # Filter the data
    we_need = plot_axis_and_grouping[:]
    we_need.append(we_plot)
    plot_data = raw_data[we_need]

    if plot_data.empty:
        print("WARINING: Empty plot !")
        return None

    hue_order = []
    for tool_name in tools:
        if tool_name in plot_data["Tool"].unique():
            hue_order.append(tool_name)

    rq_id = "RQ2"
    for the_map in plot_data["Features Combination"].unique():
        # Filter the maps first by "Features Combination" and the invoke the regular _log_statistics !
        print("============================================================================")
        print("DATASET %s Showing comparisons for MAP %s : " %(rq_id, the_map))
        print("============================================================================")

        stats_data = plot_data[plot_data["Features Combination"] == the_map]

        _log_statistics(stats_data, we_plot)

    # Return the axis to allow for additional changes
    return sns.boxplot(x="Features Combination",
                     y=we_plot,
                     hue="Tool",
                     data=plot_data,
                     palette=palette,
                     hue_order=hue_order,
                     ax=use_ax)

PAPER_FOLDER="./plots"

def create_the_table(df, file_name):

    file_format = 'txt'
    txt_file_name = "".join([file_name, ".", file_format])
    txt_file = os.path.join(PAPER_FOLDER, txt_file_name)

    with open(txt_file, "w") as text_file1:
        text_file1.write(df[["Tool", "Features Combination", "test input count"]].groupby(["Features Combination", "Tool"]).mean().to_string())
        text_file1.write("\n-------------------------------------------------------------------------------\n")
        text_file1.write(df[["Tool", "Features Combination", "num tsne clusters"]].groupby(["Features Combination", "Tool"]).mean().to_string())
        text_file1.write("\n-------------------------------------------------------------------------------\n")
        text_file1.write(df[["Tool", "Features Combination", "test input count in target"]].groupby(["Features Combination", "Tool"]).mean().to_string())
        text_file1.write("\n-------------------------------------------------------------------------------\n")
        text_file1.write(df[["Tool", "Features Combination", "target num tsne clusters"]].groupby(["Features Combination", "Tool"]).mean().to_string())

def create_the_table_retrain(df, file_name):

    file_format = 'txt'
    txt_file_name = "".join([file_name, ".", file_format])
    txt_file = os.path.join(PAPER_FOLDER, txt_file_name)

    with open(txt_file, "w") as text_file1:
        text_file1.write(df[["Tool", "Features Combination", "accuracy test set"]].groupby(["Features Combination", "Tool"]).mean().to_string())
        text_file1.write("\n-------------------------------------------------------------------------------\n")
        text_file1.write(df[["Tool", "Features Combination", "accuracy target test set"]].groupby(["Features Combination", "Tool"]).mean().to_string())
        text_file1.write("\n-------------------------------------------------------------------------------\n")



def store_figure_to_paper_folder(figure, file_name):
    import os
    try:
        os.makedirs(PAPER_FOLDER)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    figure.tight_layout()
    figure.savefig(figure_file, format=file_format, bbox_inches='tight')

    print("Plot stored to ", figure_file)
