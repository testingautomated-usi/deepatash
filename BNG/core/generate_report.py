import sys
import os
from pathlib import Path
import json
import glob
import numpy as np
import random
import csv
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
path = Path(os.path.abspath(__file__))

sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))
import core.utils as us
from core.feature_dimension import FeatureDimension
from core.plot_utils import plot_heatmap, plot_heatmap_rescaled, plot_archive_rescaled, plot_heatmap_rescaled_expansion
import math
from scipy import stats
from matplotlib.pyplot import boxplot


def eff_size_label(eff_size):
    if np.abs(eff_size) < 0.2:
        return 'negligible'
    if np.abs(eff_size) < 0.5:
        return 'small'
    if np.abs(eff_size) < 0.8:
        return 'medium'
    return 'large'

def generate_detailed_rescaled_reports(filename, log_dir_name):
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')
    # write the header
    cf.writerow(["Features", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density", "Filled Sparseness", "Misbehaviour Sparseness"])

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_MeanLateralPosition_MinRadius" in f]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    misbehaviour_sparseness = []
    filled_sparseness = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            misbehaviour_sparseness.append(float(data["Misbehaviour sparseness"]))
            filled_sparseness.append(float(data["Filled sparseness"]))
            cf.writerow(["MeanLateralPosition,MinRadius", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"] ])

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_DirectionCoverage_MinRadius" in g]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["DirectionCoverage,MinRadius", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"] ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_DirectionCoverage" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["MeanLateralPosition,DirectionCoverage", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_MinRadius" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["SegmentCount,MinRadius", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"] ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SegmentCount" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["MeanLateralPosition,SegmentCount", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_DirectionCoverage" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["SegmentCount,DirectionCoverage", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"]])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_DirectionCoverage_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["DirectionCoverage,SDSteeringAngle", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"] , data["Filled Sparsness"], data["Misbehaviour Sparsness"] ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MinRadius_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["MinRadius,SDSteeringAngle", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["SegmentCount,SDSteeringAngle", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_Curvature_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["Curvature,SDSteeringAngle", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_Curvature_MeanLateralPosition" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["Curvature,MeanLateralPosition", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_Curvature_SegmentCount" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["Curvature,SegmentCount", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"], data["Filled Sparsness"], data["Misbehaviour Sparsness"]  ])


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
            cf.writerow(["MeanLateralPosition,SDSteeringAngle", data["Filled cells"], data["Filled density"], data["Misbehaviour"], data["Misbehaviour density"] , data["Filled Sparsness"], data["Misbehaviour Sparsness"] ])

def measure_stats(filename, log_dir_name, i):
    n_sqrt = math.sqrt(i)

    meanfile = filename + "_mean.csv"
    fw = open(meanfile, 'w')
    cf = csv.writer(fw, lineterminator='\n')
    # write the header
    cf.writerow(["Features", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density"])

    semfile = filename + "_sem.csv"
    fw = open(semfile, 'w')
    cs = csv.writer(fw, lineterminator='\n')
    # write the header
    cs.writerow(["Features", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density"])

    sempfile = filename + "_sem_percent.csv"
    fw = open(sempfile, 'w')
    cp = csv.writer(fw, lineterminator='\n')
    # write the header
    cp.writerow(["Features", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density"])

    metrics = []

    filled_values = []
    filled_density_values = []
    misbehaviour_values = []
    misbehaviour_density_values = []



    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_DirectionCoverage_MinRadius" in g]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []

    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"])) 
    
    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    filled_std = np.std(filled_cells)
    filled_density_std = np.std(filled_density)
    misbehaviour_std = np.std(misbehaviour)
    misbehaviour_density_std = np.std(misbehaviour_density)

    cf.writerow(["DirectionCoverage,MinRadius", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["DirectionCoverage,MinRadius", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["DirectionCoverage,MinRadius", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])



    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_MeanLateralPosition_MinRadius" in f]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    metrics = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))

    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    filled_std = np.std(filled_cells)
    filled_density_std = np.std(filled_density)
    misbehaviour_std = np.std(misbehaviour)
    misbehaviour_density_std = np.std(misbehaviour_density)

    cf.writerow(["MeanLateralPosition,MinRadius", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["MeanLateralPosition,MinRadius", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["MeanLateralPosition,MinRadius", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])


    
    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_DirectionCoverage" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    
    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["MeanLateralPosition,DirectionCoverage", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["MeanLateralPosition,DirectionCoverage", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["MeanLateralPosition,DirectionCoverage", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_MinRadius" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))

    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["SegmentCount,MinRadius", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["SegmentCount,MinRadius", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["SegmentCount,MinRadius", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SegmentCount" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    
    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["MeanLateralPosition,SegmentCount", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["MeanLateralPosition,SegmentCount", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["MeanLateralPosition,SegmentCount", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_DirectionCoverage" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))
    
    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["SegmentCount,DirectionCoverage", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["SegmentCount,DirectionCoverage", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["SegmentCount,DirectionCoverage", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))

    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["SegmentCount,SDSteeringAngle", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["SegmentCount,SDSteeringAngle", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["SegmentCount,SDSteeringAngle", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])
       


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_DirectionCoverage_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))

    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["DirectionCoverage,SDSteeringAngle", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["DirectionCoverage,SDSteeringAngle", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["DirectionCoverage,SDSteeringAngle", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])
        
    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MinRadius_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))

    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["MinRadius,SDSteeringAngle", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["MinRadius,SDSteeringAngle", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["MinRadius,SDSteeringAngle", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])
               

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SDSteeringAngle" in h]
    filled_cells = []
    filled_density = []
    misbehaviour = []
    misbehaviour_density = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            filled_cells.append(float(data["Filled cells"]))
            filled_density.append(float(data["Filled density"]))
            misbehaviour.append(float(data["Misbehaviour"]))
            misbehaviour_density.append(float(data["Misbehaviour density"]))

    filled_values.append(filled_cells)
    filled_density_values.append(filled_density)
    misbehaviour_values.append(misbehaviour)
    misbehaviour_density_values.append(misbehaviour_density)

    filled_mean = np.mean(filled_cells)
    filled_density_mean = np.mean(filled_density)
    misbehaviour_mean = np.mean(misbehaviour)
    misbehaviour_density_mean = np.mean(misbehaviour_density)

    cf.writerow(["MeanLateralPosition,SDSteeringAngle", str(filled_mean), str(filled_density_mean), str(misbehaviour_mean), str(misbehaviour_density_mean) ])
    cs.writerow(["MeanLateralPosition,SDSteeringAngle", str(filled_std/n_sqrt), str(filled_density_std/n_sqrt), str(misbehaviour_std/n_sqrt), str(misbehaviour_density_std/n_sqrt)])
    cp.writerow(["MeanLateralPosition,SDSteeringAngle", str(((filled_std/n_sqrt)/filled_density_mean)*100), str(((filled_density_std/n_sqrt)/filled_density_mean)*100), str(((misbehaviour_std/n_sqrt)/misbehaviour_density_mean)*100), str(((misbehaviour_density_std/n_sqrt)/misbehaviour_density_mean)*100)])

    metrics = [filled_values, filled_density_values, misbehaviour_values, misbehaviour_density_values]
    return metrics 

# with numbers on map
def generate_rescaled_maps_with_archive(path, paths):
    max_MeanLateralPosition, max_SegmentCount, max_SDSteeringAngle, max_Curvature, min_MeanLateralPosition, min_SegmentCount, min_SDSteeringAngle, min_Curvature = overall_min_max(path)
    for path in paths:
        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "results_Curvature_MeanLateralPosition" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=data["MeanLateralPosition_max"])
                fts.append(ft3)

                ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=data["Curvature_max"])
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition)
                archive = us.new_rescale_archive(fts, np.array(data["Archive"]), min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition)

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition, savefig_path=path)
                plot_archive_rescaled(performances, archive,fts[1],fts[0], min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition, savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist(), 
                    'Archive': archive.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "results_Curvature_SegmentCount" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=data["SegmentCount_max"])
                fts.append(ft3)

                ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=data["Curvature_max"])
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount)
                archive = us.new_rescale_archive(fts, np.array(data["Archive"]), min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount)

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount, savefig_path=path)
                plot_archive_rescaled(performances, archive,fts[1],fts[0], min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount, savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist(),
                    'Archive': archive.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SegmentCount" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=data["SegmentCount_max"])
                fts.append(ft3)

                ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=data["MeanLateralPosition_max"])
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount)
                archive = us.new_rescale_archive(fts, np.array(data["Archive"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount)

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount, savefig_path=path)
                plot_archive_rescaled(performances, archive,fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount, savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist(),
                    'Archive': archive.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "results_Curvature_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=data["SDSteeringAngle_max"])
                fts.append(ft1)

                ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=data["Curvature_max"])
                fts.append(ft3)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle)
                archive = us.new_rescale_archive(fts, np.array(data["Archive"]), min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                plot_archive_rescaled(performances, archive,fts[1],fts[0], min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist(),
                    'Archive': archive.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=data["SDSteeringAngle_max"])
                fts.append(ft1)

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=data["SegmentCount_max"])
                fts.append(ft3)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle)
                archive = us.new_rescale_archive(fts, np.array(data["Archive"]), min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                plot_archive_rescaled(performances, archive,fts[1],fts[0], min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist(),
                    'Archive': archive.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=data["SDSteeringAngle_max"])
                fts.append(ft1)

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=data["MeanLateralPosition_max"])
                fts.append(ft3)

                
                performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle)
                archive = us.new_rescale_archive(fts, np.array(data["Archive"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                plot_archive_rescaled(performances, archive, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))

                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist(),
                    'Archive': archive.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        generate_detailed_rescaled_reports(path.replace("/", "_").replace(":", "_"), path)

def overall_min_max(path):
    max_MeanLateralPosition = 0
    max_SegmentCount = 0
    max_SDSteeringAngle = 0
    max_Curvature= 0

    min_MeanLateralPosition = np.inf
    min_SegmentCount = np.inf
    min_SDSteeringAngle = np.inf
    min_Curvature = np.inf

    jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_MeanLateralPosition" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["Curvature_max"] > max_Curvature:
                max_Curvature = data["Curvature_max"]
            if data["Curvature_min"] < min_Curvature:
                min_Curvature = data["Curvature_min"]

            if data["MeanLateralPosition_max"] > max_MeanLateralPosition:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            if data["MeanLateralPosition_min"] < min_MeanLateralPosition:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]

    jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_SegmentCount" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["Curvature_max"] > max_Curvature:
                max_Curvature = data["Curvature_max"]
            if data["Curvature_min"] < min_Curvature:
                min_Curvature = data["Curvature_min"]

            if data["SegmentCount_max"] > max_SegmentCount:
                max_SegmentCount = data["SegmentCount_max"]
            if data["SegmentCount_min"] < min_SegmentCount:
                min_SegmentCount = data["SegmentCount_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SegmentCount" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["SegmentCount_max"] > max_SegmentCount:
                max_SegmentCount = data["SegmentCount_max"]
            if data["SegmentCount_min"] < min_SegmentCount:
                min_SegmentCount = data["SegmentCount_min"]
            
            if data["MeanLateralPosition_max"] > max_MeanLateralPosition:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            if data["MeanLateralPosition_min"] < min_MeanLateralPosition:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_Curvature_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["Curvature_max"] > max_Curvature:
                max_Curvature = data["Curvature_max"]
            if data["Curvature_min"] < min_Curvature:
                min_Curvature = data["Curvature_min"]

            if data["SDSteeringAngle_max"] > max_SDSteeringAngle:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]
            if data["SDSteeringAngle_min"] < min_SDSteeringAngle:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["SegmentCount_max"] > max_SegmentCount:
                max_SegmentCount = data["SegmentCount_max"]
            if data["SegmentCount_min"] < min_SegmentCount:
                min_SegmentCount = data["SegmentCount_min"]

            if data["SDSteeringAngle_max"] > max_SDSteeringAngle:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]
            if data["SDSteeringAngle_min"] < min_SDSteeringAngle:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["MeanLateralPosition_max"] > max_MeanLateralPosition:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            if data["MeanLateralPosition_min"] < min_MeanLateralPosition:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]

            if data["SDSteeringAngle_max"] > max_SDSteeringAngle:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]
            if data["SDSteeringAngle_min"] < min_SDSteeringAngle:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]

    return max_MeanLateralPosition, max_SegmentCount, max_SDSteeringAngle, max_Curvature, min_MeanLateralPosition, min_SegmentCount, min_SDSteeringAngle, min_Curvature

def generate_rescaled_maps(path, paths):
    max_MeanLateralPosition, max_SegmentCount, max_SDSteeringAngle, max_Curvature, min_MeanLateralPosition, min_SegmentCount, min_SDSteeringAngle, min_Curvature = overall_min_max(path)
    for path in paths:
        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_MeanLateralPosition" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=data["MeanLateralPosition_max"])
                fts.append(ft3)

                ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=data["Curvature_max"])
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_SegmentCount" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=data["SegmentCount_max"])
                fts.append(ft3)

                ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=data["Curvature_max"])
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount)
                
                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SegmentCount" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=data["SegmentCount_max"])
                fts.append(ft3)

                ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=data["MeanLateralPosition_max"])
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_Curvature_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=data["SDSteeringAngle_max"])
                fts.append(ft1)

                ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=data["Curvature_max"])
                fts.append(ft3)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=data["SDSteeringAngle_max"])
                fts.append(ft1)

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=data["SegmentCount_max"])
                fts.append(ft3)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=data["SDSteeringAngle_max"])
                fts.append(ft1)

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=data["MeanLateralPosition_max"])
                fts.append(ft3)

                
                performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                
                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))

                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        generate_detailed_rescaled_reports(path.replace("/", "_").replace(":", "_"), path)

def generate_rescaled_maps_by_report(path, paths):
    max_MeanLateralPosition = 0
    max_SegmentCount = 0
    max_SDSteeringAngle = 0
    max_Curvature= 0

    min_MeanLateralPosition = np.inf
    min_SegmentCount = np.inf
    min_SDSteeringAngle = np.inf
    min_Curvature = np.inf

    jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_MeanLateralPosition" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["Curvature_max"] > max_Curvature:
                max_Curvature = data["Curvature_max"]
            if data["Curvature_min"] < min_Curvature:
                min_Curvature = data["Curvature_min"]

            if data["MeanLateralPosition_max"] > max_MeanLateralPosition:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            if data["MeanLateralPosition_min"] < min_MeanLateralPosition:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]

    jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_SegmentCount" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["Curvature_max"] > max_Curvature:
                max_Curvature = data["Curvature_max"]
            if data["Curvature_min"] < min_Curvature:
                min_Curvature = data["Curvature_min"]

            if data["SegmentCount_max"] > max_SegmentCount:
                max_SegmentCount = data["SegmentCount_max"]
            if data["SegmentCount_min"] < min_SegmentCount:
                min_SegmentCount = data["SegmentCount_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SegmentCount" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["SegmentCount_max"] > max_SegmentCount:
                max_SegmentCount = data["SegmentCount_max"]
            if data["SegmentCount_min"] < min_SegmentCount:
                min_SegmentCount = data["SegmentCount_min"]
            
            if data["MeanLateralPosition_max"] > max_MeanLateralPosition:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            if data["MeanLateralPosition_min"] < min_MeanLateralPosition:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_Curvature_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["Curvature_max"] > max_Curvature:
                max_Curvature = data["Curvature_max"]
            if data["Curvature_min"] < min_Curvature:
                min_Curvature = data["Curvature_min"]

            if data["SDSteeringAngle_max"] > max_SDSteeringAngle:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]
            if data["SDSteeringAngle_min"] < min_SDSteeringAngle:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["SegmentCount_max"] > max_SegmentCount:
                max_SegmentCount = data["SegmentCount_max"]
            if data["SegmentCount_min"] < min_SegmentCount:
                min_SegmentCount = data["SegmentCount_min"]

            if data["SDSteeringAngle_max"] > max_SDSteeringAngle:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]
            if data["SDSteeringAngle_min"] < min_SDSteeringAngle:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]

    jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if data["MeanLateralPosition_max"] > max_MeanLateralPosition:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            if data["MeanLateralPosition_min"] < min_MeanLateralPosition:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]

            if data["SDSteeringAngle_max"] > max_SDSteeringAngle:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]
            if data["SDSteeringAngle_min"] < min_SDSteeringAngle:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]

    for path in paths:
        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_MeanLateralPosition" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "report_Curvature_SegmentCount" in f]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
                fts.append(ft3)

                ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount)
                
                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SegmentCount" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_Curvature_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)

                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, np.array(data["Performances"]), min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))
                                            
                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SDSteeringAngle" in h]
        for json_data in jsons:
            with open(json_data) as json_file:
                data = json.load(json_file)
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                
                performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle)
                

                plot_heatmap_rescaled(performances, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)
                
                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0
                filled_dists = []
                filled2 = []
                missed = []
                missed_dists = []

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                        filled2.append((i,j))
                        if performances[i, j] < 0:
                            COUNT_MISS += 1
                            missed.append((i,j))

                for ind in filled2:
                    filled_dists.append(get_max_distance_from_set(ind, filled2))

                for ind in missed:
                    missed_dists.append(get_max_distance_from_set(ind, missed))

                if len(filled2) > 0:
                    filled_sp = sum(filled_dists)/len(filled2)
                else:
                    filled_sp = 0
                if len(missed) > 0:
                    missed_sp = sum(missed_dists)/len(missed)
                else:
                    missed_sp = 0
                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled),
                    'Misbehaviour Sparsness': str(missed_sp),
                    'Filled Sparsness': str(filled_sp),
                    'Performances': performances.tolist()

                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        generate_detailed_rescaled_reports(path.replace("/", "_").replace(":", "_"), path)

def get_max_distance_from_set(ind, solution):
    distances = list()
    # print("ind:", ind)
    # print("solution:", solution)
    ind_spine = ind

    for road in solution:
        road_spine = road
        distances.append(manhattan_dist(ind_spine, road_spine))
    distances.sort()
    return distances[-1]

def manhattan_dist(ind1, ind2):
    return abs(ind1[0] - ind2[0]) + abs(ind1[1] - ind2[1])

def calculate_effect_size(paths):
    metrics = []
    for path in paths:
        metric = measure_stats(path.replace("/", "_"), path, 10)
        metrics.append(metric)


    dh = metrics[0]
    dj = metrics[1]
    af = metrics[2]

    feature_combination = ["DirectionCoverage, MinRadius", "MeanLateralPosition, MinRadius", "MeanLateralPosition, DirectionCoverage", 
    "SegmentCount, MinRadius", "MeanLateralPosition, SegmentCount", "SegmentCount, DirectionCoverage", "SegmentCount, SDSteeringAngle", 
    "DirectionCoverage, SDSteeringAngle", "MinRadius, SDSteeringAngle", "MeanLateralPosition, SDSteeringAngle"]
    metric_label = ["filled cells", "filled density", "misbehavior", "misbehavior density"]


    with open("dhvsdj.txt", "w") as text_file1:
        with open("dhvsaf.txt", "w") as text_file2:
            # metrics
            text_file1.write("DeepHyperion vs DeepJanus\n")
            text_file2.write("DeepHyperion vs AsFault\n")
            for k in range(0,4):
                text_file1.write(metric_label[k]+ "\n")
                text_file2.write(metric_label[k]+ "\n")
                # feature combinations
                for i in range(0, 10):
                    boxplot([dh[k][i], dj[k][i]], labels=["DeepHyperion", "DeepJanus"])
                    boxplot([dh[k][i], af[k][i]], labels=["DeepHyperion", "AsFault"])

                    (t, p) = stats.wilcoxon(dh[k][i], dj[k][i])
                    eff_size = (np.mean(dh[k][i]) - np.mean(dj[k][i])) / np.sqrt((np.std(dh[k][i]) ** 2 + np.std(dj[k][i]) ** 2) / 2.0)
                    
                    text_file1.write(f"{feature_combination[i]}: Cohen effect size = {eff_size} ({eff_size_label(eff_size)} ); Wilcoxon p-value =  {p}\n")
                    
                    (t, p) = stats.wilcoxon(dh[k][i], af[k][i])
                    eff_size = (np.mean(dh[k][i]) - np.mean(af[k][i])) / np.sqrt((np.std(dh[k][i]) ** 2 + np.std(af[k][i]) ** 2) / 2.0)
                    text_file2.write(f"{feature_combination[i]}: Cohen effect size = {eff_size} ({eff_size_label(eff_size)}); Wilcoxon p-value = {p}\n")

def calculate_map_differences(DH_Path, Training_Paths):
    max_MeanLateralPosition, max_SegmentCount, max_SDSteeringAngle, max_Curvature, min_MeanLateralPosition, min_SegmentCount, min_SDSteeringAngle, min_Curvature = overall_min_max(DH_Path)
    for path in Training_Paths:

        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_Curvature_MeanLateralPosition" in f]
        dh_json = [f for f in sorted(glob.glob(f"{DH_Path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_Curvature_MeanLateralPosition" in f]
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        with open(dh_json[0]) as json_file1:
            dh_data = json.load(json_file1)


        fts = list()

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

        ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
        fts.append(ft1)

        dh_performances = np.array(dh_data["Performances"])
        other_performances = np.array(data["Performances"])
        archive = np.array(data["Archive"])

        filled_diff = []
        missed_diff = []
        density = 0
        for (i, j), value in np.ndenumerate(dh_performances):
            if dh_performances[i, j] != np.inf:
                if other_performances[i,j] == np.inf:
                    filled_diff.append((j,i))
                    if dh_performances[i, j] < 0:
                        missed_diff.append((i,j))
                elif dh_performances[i, j] < 0:
                    density += archive[i,j]
            
                
        plot_heatmap_rescaled_expansion(dh_performances, filled_diff, fts[1],fts[0], min_Curvature, max_Curvature, min_MeanLateralPosition, max_MeanLateralPosition, savefig_path=path)        

        other_filled = []
        for (i, j), value in np.ndenumerate(other_performances):
            if other_performances[i,j] != np.inf:
                other_filled.append((i,j))
                

        filled = len(filled_diff)
        missed = len(missed_diff)

                
        report = {
            'Training Filled': str(len(other_filled)),
            'Filled expansion': str(filled),
            'Misbehaviour expansion': str(missed),
            'Density': str(density)
        }

        dst = f"{path}/difference_" + fts[1].name + "_" + fts[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()


        jsons = [f for f in sorted(glob.glob(f"{path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_Curvature_SegmentCount" in f]
        dh_json = [f for f in sorted(glob.glob(f"{DH_Path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_Curvature_SegmentCount" in f]
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        with open(dh_json[0]) as json_file:
            dh_data = json.load(json_file)


        fts = list()

        ft1 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft1)

        ft3 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
        fts.append(ft3)

        dh_performances = np.array(dh_data["Performances"])
        other_performances = np.array(data["Performances"])

        archive = np.array(data["Archive"])

        filled_diff = []
        missed_diff = []
        density = 0
        for (i, j), value in np.ndenumerate(dh_performances):
            if dh_performances[i, j] != np.inf:
                if other_performances[i,j] == np.inf:
                    filled_diff.append((j,i))
                    if dh_performances[i, j] < 0:
                        missed_diff.append((i,j))
                elif dh_performances[i, j] < 0:
                    density += archive[i,j]

        other_filled = []
        for (i, j), value in np.ndenumerate(other_performances):
            if other_performances[i,j] != np.inf:
                other_filled.append((i,j))
                
        plot_heatmap_rescaled_expansion(dh_performances, filled_diff, fts[1],fts[0], min_Curvature, max_Curvature, min_SegmentCount, max_SegmentCount, savefig_path=path)

        filled = len(filled_diff)
        missed = len(missed_diff)

                
        report = {
            'Training Filled': str(len(other_filled)),
            'Filled expansion': str(filled),
            'Misbehaviour expansion': str(missed),
            'Density': str(density)
        }

        dst = f"{path}/difference_" + fts[1].name + "_" + fts[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()

        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SegmentCount" in h]
        dh_json = [f for f in sorted(glob.glob(f"{DH_Path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_MeanLateralPosition_SegmentCount" in f]
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        with open(dh_json[0]) as json_file:
            dh_data = json.load(json_file)


        fts = list()

        ft1 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft1)

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

        dh_performances = np.array(dh_data["Performances"])
        other_performances = np.array(data["Performances"])

        archive = np.array(data["Archive"])

        filled_diff = []
        missed_diff = []
        density = 0
        for (i, j), value in np.ndenumerate(dh_performances):
            if dh_performances[i, j] != np.inf:
                if other_performances[i,j] == np.inf:
                    filled_diff.append((j,i))
                    if dh_performances[i, j] < 0:
                        missed_diff.append((i,j))
                elif dh_performances[i, j] < 0:
                    density += archive[i,j]

        other_filled = []
        for (i, j), value in np.ndenumerate(other_performances):
            if other_performances[i,j] != np.inf:
                other_filled.append((i,j))
                
        plot_heatmap_rescaled_expansion(dh_performances, filled_diff,  fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount, savefig_path=path)

        filled = len(filled_diff)
        missed = len(missed_diff)

                
        report = {
            'Training Filled': str(len(other_filled)),
            'Filled expansion': str(filled),
            'Misbehaviour expansion': str(missed),
            'Density': str(density)
        }

        dst = f"{path}/difference_" + fts[1].name + "_" + fts[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()


        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_Curvature_SDSteeringAngle" in h]
        dh_json = [f for f in sorted(glob.glob(f"{DH_Path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_Curvature_SDSteeringAngle" in f]
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        with open(dh_json[0]) as json_file:
            dh_data = json.load(json_file)


        fts = list()

        ft3 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering_angle", bins=1)
        fts.append(ft3)

        ft1 = FeatureDimension(name="Curvature", feature_simulator="curvature", bins=1)
        fts.append(ft1)

        dh_performances = np.array(dh_data["Performances"])
        other_performances = np.array(data["Performances"])

        archive = np.array(data["Archive"])

        filled_diff = []
        missed_diff = []
        density = 0
        for (i, j), value in np.ndenumerate(dh_performances):
            if dh_performances[i, j] != np.inf:
                if other_performances[i,j] == np.inf:
                    filled_diff.append((j,i))
                    if dh_performances[i, j] < 0:
                        missed_diff.append((i,j))
                elif dh_performances[i, j] < 0:
                    density += archive[i,j]

        other_filled = []
        for (i, j), value in np.ndenumerate(other_performances):
            if other_performances[i,j] != np.inf:
                other_filled.append((i,j))
                
        plot_heatmap_rescaled_expansion(dh_performances, filled_diff, fts[1],fts[0], min_Curvature, max_Curvature, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)

        filled = len(filled_diff)
        missed = len(missed_diff)

                
        report = {
            'Training Filled': str(len(other_filled)),
            'Filled expansion': str(filled),
            'Misbehaviour expansion': str(missed),
            'Density': str(density)
        }

        dst = f"{path}/difference_" + fts[1].name + "_" + fts[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()



        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_SDSteeringAngle" in h]
        dh_json = [f for f in sorted(glob.glob(f"{DH_Path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_SegmentCount_SDSteeringAngle" in f]
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        with open(dh_json[0]) as json_file:
            dh_data = json.load(json_file)


        fts = list()

        ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering_angle", bins=1)
        fts.append(ft1)

        ft3 = FeatureDimension(name="SegmentCount", feature_simulator="segment_count", bins=1)
        fts.append(ft3)

        dh_performances = np.array(dh_data["Performances"])
        other_performances = np.array(data["Performances"])

        archive = np.array(data["Archive"])

        filled_diff = []
        missed_diff = []
        density = 0
        for (i, j), value in np.ndenumerate(dh_performances):
            if dh_performances[i, j] != np.inf:
                if other_performances[i,j] == np.inf:
                    filled_diff.append((j,i))
                    if dh_performances[i, j] < 0:
                        missed_diff.append((i,j))
                elif dh_performances[i, j] < 0:
                    density += archive[i,j]
                

        other_filled = []
        for (i, j), value in np.ndenumerate(other_performances):
            if other_performances[i,j] != np.inf:
                other_filled.append((i,j))
                
        plot_heatmap_rescaled_expansion(dh_performances, filled_diff, fts[1],fts[0], min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path)

        filled = len(filled_diff)
        missed = len(missed_diff)

                
        report = {
            'Training Filled': str(len(other_filled)),
            'Filled expansion': str(filled),
            'Misbehaviour expansion': str(missed),
            'Density': str(density)
        }

        dst = f"{path}/difference_" + fts[1].name + "_" + fts[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()



        jsons = [h for h in sorted(glob.glob(f"{path}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SDSteeringAngle" in h]
        dh_json = [f for f in sorted(glob.glob(f"{DH_Path}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_MeanLateralPosition_SDSteeringAngle" in f]
        with open(jsons[0]) as json_file:
            data = json.load(json_file)
        with open(dh_json[0]) as json_file:
            dh_data = json.load(json_file)


        fts = list()

        ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="sd_steering_angle", bins=1)
        fts.append(ft1)

        ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
        fts.append(ft3)

        dh_performances = np.array(dh_data["Performances"])
        other_performances = np.array(data["Performances"])

        

        archive = np.array(data["Archive"])

        filled_diff = []
        missed_diff = []
        density = 0
        for (i, j), value in np.ndenumerate(dh_performances):
            if dh_performances[i, j] != np.inf:
                if other_performances[i,j] == np.inf:
                    filled_diff.append((j,i))
                    if dh_performances[i, j] < 0:
                        missed_diff.append((i,j))
                elif dh_performances[i, j] < 0:
                    density += archive[i,j]

        other_filled = []
        for (i, j), value in np.ndenumerate(other_performances):
            if other_performances[i,j] != np.inf:
                other_filled.append((i,j))

        plot_heatmap_rescaled_expansion(dh_performances, filled_diff, fts[1],fts[0], min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle, savefig_path=path) 

        filled = len(filled_diff)
        missed = len(missed_diff)

                
        report = {
            'Training Filled': str(len(other_filled)),
            'Filled expansion': str(filled),
            'Misbehaviour expansion': str(missed),
            'Density': str(density)
        }

        dst = f"{path}/difference_" + fts[1].name + "_" + fts[
            0].name + '.json'
        report_string = json.dumps(report)

        file = open(dst, 'w')
        file.write(report_string)
        file.close()

def test_box_plot():
    tool_colors = {
        "DeepHyperion": "#ffffff",
        "DeepJanus" : "#d3d3d3", #C0C0C0 - #DCDCDC
        "AsFault": "#a9a9a9" # #808080
    }
    file_name = "Mapped Misbehaviour"
    fig, ax = plt.subplots(figsize=(10, 8))
    
    data = pd.read_csv("report-df.csv")
    ax = sns.boxplot(x="Feature Combination", y="Mapped Misbehaviour", hue="Tool",data=data, palette="Set3")
    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    #figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.savefig(figure_file_name, format=file_format, bbox_inches='tight')

    file_name = "Coverage"
    fig, ax = plt.subplots(figsize=(10, 8))
    

    ax = sns.boxplot(x="Feature Combination", y="Coverage", hue="Tool",data=data, palette="Set3")
    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    #figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.savefig(figure_file_name, format=file_format, bbox_inches='tight')

    file_name = "Misbehaviour Sparseness"
    fig, ax = plt.subplots(figsize=(10, 8))
    

    ax = sns.boxplot(x="Feature Combination", y="Misbehaviour Sparseness", hue="Tool",data=data, palette="Set3")
    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    #figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.savefig(figure_file_name, format=file_format, bbox_inches='tight')

    file_name = "Coverage Sparseness"
    fig, ax = plt.subplots(figsize=(10, 8))
    

    ax = sns.boxplot(x="Feature Combination", y="Coverage Sparseness", hue="Tool",data=data, palette="Set3")
    file_format = 'pdf'
    figure_file_name = "".join([file_name, ".", file_format])
    #figure_file = os.path.join(PAPER_FOLDER, figure_file_name)

    # https://stackoverflow.com/questions/4042192/reduce-left-and-right-margins-in-matplotlib-plot
    fig.tight_layout()
    fig.savefig(figure_file_name, format=file_format, bbox_inches='tight')


def generate_csvs_training(filename, log_dir_name):
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')
    # write the header
    cf.writerow(["Features", "Filled Cells", "Filled Cells Only DH",
                 "Misbehaviors Only DH", "Filled Cells Only Training", "Misbehavior Only Training",
                 "Filled Cells Both", "Misbehaviors DH Filled Training"])

    # jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "MeanLateralPosition-Curvature-white-box-rescaled-stats" in f]
    # Filled_Cells = []
    # Filled_Cells_Only_DH = []
    # Misbehaviors_Only_DH  = []
    # Filled_Cells_Only_Training = []
    # Misbehavior_Only_Training = []
    # Filled_Cells_Both = []
    # Misbehaviors_DH_Filled_Training = []
    # for json_data in jsons:
    #     with open(json_data) as json_file:
    #         data = json.load(json_file)["Reports"][0]
    #         Filled_Cells.append(float(data["Filled Cells"]))
    #         Filled_Cells_Only_DH.append(float(data["Filled Cells Only DH"]))
    #         Misbehaviors_Only_DH.append(float(data["Misbehaviors Only DH"]))
    #         Filled_Cells_Only_Training.append(float(data["Filled Cells Only Training"]))
    #         Misbehavior_Only_Training.append(float(data["Misbehavior Only Training"]))
    #         Filled_Cells_Both.append(float(data["Filled Cells Both"]))
    #         Misbehaviors_DH_Filled_Training.append(float(data["Misbehaviors DH Filled Training"]))
    # cf.writerow(["MeanLateralPosition, Curvature",str(format(np.mean(Filled_Cells),'.2f')), str(format(np.mean(Filled_Cells_Only_DH),'.2f')), str(format(np.mean(Misbehaviors_Only_DH),'.2f')),  
    # str(format(np.mean(Filled_Cells_Only_Training),'.2f')),  str(format(np.mean(Misbehavior_Only_Training),'.2f')),
    # str(format(np.mean(Filled_Cells_Both),'.2f')),  str(format(np.mean(Misbehaviors_DH_Filled_Training),'.2f')) ])

    
    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "MeanLateralPosition-SegmentCount-white-box-rescaled-stats" in f]
    Filled_Cells = []
    Filled_Cells_Only_DH = []
    Misbehaviors_Only_DH  = []
    Filled_Cells_Only_Training = []
    Misbehavior_Only_Training = []
    Filled_Cells_Both = []
    Misbehaviors_DH_Filled_Training = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)["Reports"][0]
            Filled_Cells.append(float(data["Filled Cells"]))
            Filled_Cells_Only_DH.append(float(data["Filled Cells Only DH"]))
            Misbehaviors_Only_DH.append(float(data["Misbehaviors Only DH"]))
            Filled_Cells_Only_Training.append(float(data["Filled Cells Only Training"]))
            Misbehavior_Only_Training.append(float(data["Misbehavior Only Training"]))
            Filled_Cells_Both.append(float(data["Filled Cells Both"]))
            Misbehaviors_DH_Filled_Training.append(float(data["Misbehaviors DH Filled Training"]))
    cf.writerow(["MeanLateralPosition, SegmentCount",str(format(np.std(Filled_Cells),'.2f')), str(format(np.std(Filled_Cells_Only_DH),'.2f')), str(format(np.std(Misbehaviors_Only_DH),'.2f')),  
    str(format(np.std(Filled_Cells_Only_Training),'.2f')),  str(format(np.std(Misbehavior_Only_Training),'.2f')),
    str(format(np.std(Filled_Cells_Both),'.2f')),  str(format(np.std(Misbehaviors_DH_Filled_Training),'.2f')) ])

    
    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "MeanLateralPosition-SDSteeringAngle-white-box-rescaled-stats" in f]
    Filled_Cells = []
    Filled_Cells_Only_DH = []
    Misbehaviors_Only_DH  = []
    Filled_Cells_Only_Training = []
    Misbehavior_Only_Training = []
    Filled_Cells_Both = []
    Misbehaviors_DH_Filled_Training = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)["Reports"][0]
            Filled_Cells.append(float(data["Filled Cells"]))
            Filled_Cells_Only_DH.append(float(data["Filled Cells Only DH"]))
            Misbehaviors_Only_DH.append(float(data["Misbehaviors Only DH"]))
            Filled_Cells_Only_Training.append(float(data["Filled Cells Only Training"]))
            Misbehavior_Only_Training.append(float(data["Misbehavior Only Training"]))
            Filled_Cells_Both.append(float(data["Filled Cells Both"]))
            Misbehaviors_DH_Filled_Training.append(float(data["Misbehaviors DH Filled Training"]))
    cf.writerow(["MeanLateralPosition,SDSteeringAngle",str(format(np.std(Filled_Cells),'.2f')), str(format(np.std(Filled_Cells_Only_DH),'.2f')), str(format(np.std(Misbehaviors_Only_DH),'.2f')),  
    str(format(np.std(Filled_Cells_Only_Training),'.2f')),  str(format(np.std(Misbehavior_Only_Training),'.2f')),
    str(format(np.std(Filled_Cells_Both),'.2f')),  str(format(np.std(Misbehaviors_DH_Filled_Training),'.2f')) ])


    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "SDSteeringAngle-Curvature-white-box-rescaled-stats" in g]
    Filled_Cells = []
    Filled_Cells_Only_DH = []
    Misbehaviors_Only_DH  = []
    Filled_Cells_Only_Training = []
    Misbehavior_Only_Training = []
    Filled_Cells_Both = []
    Misbehaviors_DH_Filled_Training = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)["Reports"][0]
            Filled_Cells.append(float(data["Filled Cells"]))
            Filled_Cells_Only_DH.append(float(data["Filled Cells Only DH"]))
            Misbehaviors_Only_DH.append(float(data["Misbehaviors Only DH"]))
            Filled_Cells_Only_Training.append(float(data["Filled Cells Only Training"]))
            Misbehavior_Only_Training.append(float(data["Misbehavior Only Training"]))
            Filled_Cells_Both.append(float(data["Filled Cells Both"]))
            Misbehaviors_DH_Filled_Training.append(float(data["Misbehaviors DH Filled Training"]))
    cf.writerow(["SDSteeringAngle, Curvature", str(format(np.std(Filled_Cells),'.2f')), str(format(np.std(Filled_Cells_Only_DH),'.2f')), str(format(np.std(Misbehaviors_Only_DH),'.2f')),  
    str(format(np.std(Filled_Cells_Only_Training),'.2f')),  str(format(np.std(Misbehavior_Only_Training),'.2f')),
    str(format(np.std(Filled_Cells_Both),'.2f')),  str(format(np.std(Misbehaviors_DH_Filled_Training),'.2f')) ])

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "SegmentCount-Curvature-white-box-rescaled-stats" in h]
    Filled_Cells = []
    Filled_Cells_Only_DH = []
    Misbehaviors_Only_DH  = []
    Filled_Cells_Only_Training = []
    Misbehavior_Only_Training = []
    Filled_Cells_Both = []
    Misbehaviors_DH_Filled_Training = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)["Reports"][0]
            Filled_Cells.append(float(data["Filled Cells"]))
            Filled_Cells_Only_DH.append(float(data["Filled Cells Only DH"]))
            Misbehaviors_Only_DH.append(float(data["Misbehaviors Only DH"]))
            Filled_Cells_Only_Training.append(float(data["Filled Cells Only Training"]))
            Misbehavior_Only_Training.append(float(data["Misbehavior Only Training"]))
            Filled_Cells_Both.append(float(data["Filled Cells Both"]))
            Misbehaviors_DH_Filled_Training.append(float(data["Misbehaviors DH Filled Training"]))
    
    # cf.writerow(["Orientation, Moves", str(format(np.mean(Filled_Cells),'.2f')), str(format(np.mean(Filled_Cells_Only_DH),'.2f')), str(format(np.mean(Misbehaviors_Only_DH),'.2f')),  
    # str(format(np.mean(Filled_Cells_Only_Training),'.2f')),  str(format(np.mean(Misbehavior_Only_Training),'.2f')),
        # str(format(np.mean(Filled_Cells_Both),'.2f')),  str(format(np.mean(Misbehaviors_DH_Filled_Training),'.2f')) ])
    cf.writerow(["SegmentCount, Curvature", str(format(np.mean(Filled_Cells),'.2f')), str(format(np.mean(Filled_Cells_Only_DH),'.2f')), str(format(np.mean(Misbehaviors_Only_DH),'.2f')),  
    str(format(np.mean(Filled_Cells_Only_Training),'.2f')),  str(format(np.mean(Misbehavior_Only_Training),'.2f')),
    str(format(np.mean(Filled_Cells_Both),'.2f')),  str(format(np.mean(Misbehaviors_DH_Filled_Training),'.2f')) ])

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "SegmentCount-SDSteeringAngle-white-box-rescaled-stats" in f]
    Filled_Cells = []
    Filled_Cells_Only_DH = []
    Misbehaviors_Only_DH  = []
    Filled_Cells_Only_Training = []
    Misbehavior_Only_Training = []
    Filled_Cells_Both = []
    Misbehaviors_DH_Filled_Training = []
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)["Reports"][0]
            Filled_Cells.append(float(data["Filled Cells"]))
            Filled_Cells_Only_DH.append(float(data["Filled Cells Only DH"]))
            Misbehaviors_Only_DH.append(float(data["Misbehaviors Only DH"]))
            Filled_Cells_Only_Training.append(float(data["Filled Cells Only Training"]))
            Misbehavior_Only_Training.append(float(data["Misbehavior Only Training"]))
            Filled_Cells_Both.append(float(data["Filled Cells Both"]))
            Misbehaviors_DH_Filled_Training.append(float(data["Misbehaviors DH Filled Training"]))
    cf.writerow(["SegmentCount, SDSteeringAngle",str(format(np.mean(Filled_Cells),'.2f')), str(format(np.mean(Filled_Cells_Only_DH),'.2f')), str(format(np.mean(Misbehaviors_Only_DH),'.2f')),  
    str(format(np.mean(Filled_Cells_Only_Training),'.2f')),  str(format(np.mean(Misbehavior_Only_Training),'.2f')),
    str(format(np.mean(Filled_Cells_Both),'.2f')),  str(format(np.mean(Misbehaviors_DH_Filled_Training),'.2f')) ])


    

if __name__ == "__main__":
    # generate_rescaled_maps("New-Agent",["New-Agent/DH_logs", "New-Agent/DJ_logs", "New-Agent/New-AF"])
    # generate_rescaled_maps("C:/Users/Nabaut/Desktop/BNG/SegmentCount_MeanLateralPosition",["C:/Users/Nabaut/Desktop/BNG/SegmentCount_MeanLateralPosition/run_1", "C:/Users/Nabaut/Desktop/BNG/SegmentCount_MeanLateralPosition/run_div_1"])
    # generate_rescaled_maps("C:/Users/Nabaut/Desktop/BNG/SegmentCount_Curvature",["C:/Users/Nabaut/Desktop/BNG/SegmentCount_Curvature/run_1", "C:/Users/Nabaut/Desktop/BNG/SegmentCount_Curvature/run_div_1"])
    # generate_rescaled_maps("C:/Users/Nabaut/Desktop/BNG/SDSteeringAngle_MeanLateralPosition",["C:/Users/Nabaut/Desktop/BNG/SDSteeringAngle_MeanLateralPosition/run_1", "C:/Users/Nabaut/Desktop/BNG/SDSteeringAngle_MeanLateralPosition/run_div_1"])
    #generate_rescaled_maps("SMInitials",["SMInitials/Curvature_MeanLateralPosition","SMInitials/Curvature_SDSteeringAngle","SMInitials/Curvature_SegmentCount", "SMInitials/MeanLateralPosition_SDSteeringAngle", "SMInitials/MODIFIED-AF_logs", "SMInitials/DJ_logs"])
    #generate_rescaled_maps("New-Agent-2",["New-Agent-2/DH_logs", "New-Agent-2/DJ_logs", "New-Agent-2/MODIFIED-AF_logs"])
    # generate_rescaled_maps("Training/DH_logs",["Training/DH_logs"])
    # generate_rescaled_maps_with_archive("Training/DH_logs",["Training/All-Training_logs"])
    # calculate_map_differences("Training/DH_logs", ["Training/All-Training_logs"])
    #generate_rescaled_maps_by_report("Presentation",["Presentation/01"])
    #generate_rescaled_maps_by_report("Presentation",["Presentation/02"])
    #generate_rescaled_maps_by_report("Presentation",["Presentation/03"])
    generate_csvs_training('training_map_expansion_std_bng','D:/tara/Results/TOSEM/BNG/DeepHyperion-CS')
