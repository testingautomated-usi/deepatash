import sys
import os
from pathlib import Path
import json
import glob
import numpy as np
import random
import csv
import time
#sys.path.insert(0, r'C:\DeepHyperion-BNG')
#sys.path.append(os.path.dirname(os.path.dirname(os.path.join(__file__))))
path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))
import core.utils as us
from core.feature_dimension import FeatureDimension
from core.plot_utils import plot_heatmap
from core.mapelites_bng import MapElitesBNG
from self_driving.beamng_member import BeamNGMember
import self_driving.beamng_config as cfg
import self_driving.beamng_problem as BeamNGProblem
from self_driving.beamng_individual import BeamNGIndividual
from self_driving.beamng_road_imagery import BeamNGRoadImagery
from self_driving.simulation_data import SimulationDataRecordProperties
from self_driving.road_bbox import RoadBoundingBox
from self_driving.simulation_data import SimulationParams, SimulationDataRecords, SimulationData, SimulationDataRecord
from self_driving.vehicle_state_reader import VehicleState
from pprint import pprint

def generate_rescaled_maps_without_sim(log_dir_name, paths):
    min_MinRadius = np.inf
    min_MeanLateralPosition = np.inf
    min_DirectionCoverage = np.inf
    min_SegmentCount = np.inf
    min_SDSteeringAngle = np.inf

    max_MinRadius = 0
    max_MeanLateralPosition = 0
    max_DirectionCoverage = 0
    max_SegmentCount = 0
    max_SDSteeringAngle = 0

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "results_MeanLateralPosition_MinRadius" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "results_DirectionCoverage_MinRadius" in g]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_DirectionCoverage" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]

            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]



    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_MinRadius" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SegmentCount" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]

            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_DirectionCoverage" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]

            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_DirectionCoverage_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MinRadius_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    print("MinRadius: ", min_MinRadius, max_MinRadius)
    print("MeanLateralPosition: ", min_MeanLateralPosition, max_MeanLateralPosition)
    print("DirectionCoverage: ", min_DirectionCoverage, max_DirectionCoverage)
    print("SegmentCount: ", min_SegmentCount, max_SegmentCount)
    print("SDSteeringAngle: ", min_SDSteeringAngle, max_SDSteeringAngle)

    
    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    for path in paths:
        folders = sorted(glob.glob(path + "/*/"), key=os.path.getmtime)
        for folder in folders:
            jsons = []
            jsons = [f for f in sorted(glob.glob(f"{folder}/MeanLateralPosition_MinRadius/*.json", recursive=True),key=os.path.getmtime) if "simulation_mbr" in f]
            map_E = MapElitesBNG(1, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)
            
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = map_E.performances#us.new_rescale(fts, map_E.performances, min_MeanLateralPosition, max_MeanLateralPosition, min_MinRadius, max_MinRadius)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            

            jsons = []
            jsons = [g for g in sorted(glob.glob(f"{folder}/**/DirectionCoverage_MinRadius/*.json", recursive=True),key=os.path.getmtime) if "simulation_mbr" in g]
            map_E = MapElitesBNG(0, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_DirectionCoverage, max_DirectionCoverage, min_MinRadius, max_MinRadius)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/MeanLateralPosition_DirectionCoverage/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(2, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)

            
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_DirectionCoverage, max_DirectionCoverage, min_MeanLateralPosition, max_MeanLateralPosition)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()


            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/SegmentCount_MinRadius/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(3, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)

            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_SegmentCount, max_SegmentCount, min_MinRadius, max_MinRadius)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/MeanLateralPosition_SegmentCount/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(4, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_SegmentCount, max_SegmentCount, min_MeanLateralPosition, max_MeanLateralPosition)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/SegmentCount_DirectionCoverage/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(5, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="SegmentCount", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = map_E.performances#us.new_rescale(fts, map_E.performances, min_DirectionCoverage, max_DirectionCoverage, min_SegmentCount, max_SegmentCount)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/DirectionCoverage_SDSteeringAngle/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(7, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_DirectionCoverage, max_DirectionCoverage, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/MinRadius_SDSteeringAngle/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(8, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)
            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="MinRadius", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = map_E.performances#us.new_rescale(fts, map_E.performances, min_MinRadius, max_MinRadius, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/SegmentCount_SDSteeringAngle/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(6, problem, "", True)
            for json_data in jsons:

                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim_and_map(individual, )

            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                    0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

            jsons = []
            jsons = [h for h in sorted(glob.glob(f"{folder}/**/MeanLateralPosition_SDSteeringAngle/*.json", recursive=True), key=os.path.getmtime) if "simulation_mbr" in h]
            map_E = MapElitesBNG(9, problem, "", True)
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    control_nodes = data["road"]["nodes"]
                    for node in control_nodes:
                        node[2] = -28.0
                    control_nodes = np.array(control_nodes)
                    records = data["records"]
                    bbox_size = (-250.0, 0.0, 250.0, 500.0)
                    road_bbox = RoadBoundingBox(bbox_size)
                    member = BeamNGMember([], [tuple(t) for t in control_nodes], len(control_nodes), road_bbox)
                    member.config = config
                    member.problem = problem
                    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                    sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
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

                    simulation_data.states = states

                    if len(states) > 0:
                        member.distance_to_boundary = simulation_data.min_oob_distance()
                        member.simulation = simulation_data
                        individual: BeamNGIndividual = BeamNGIndividual(member, config)
                        map_E.place_in_mapelites_without_sim(individual)

            if len(jsons) > 0:
                fts = list()

                ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                fts.append(ft1)

                ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
                fts.append(ft3)

                performances = us.new_rescale(fts, map_E.performances, min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle)

                plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                # filled values
                total = np.size(performances)

                filled = np.count_nonzero(performances != np.inf)
                COUNT_MISS = 0

                for (i, j), value in np.ndenumerate(performances):
                    if performances[i, j] != np.inf:
                            if performances[i, j] < 0:
                                COUNT_MISS += 1

                report = {
                    'Filled cells': str(filled),
                    'Filled density': str(filled / total),
                    'Misbehaviour': str(COUNT_MISS),
                    'Misbehaviour density': str(COUNT_MISS / filled)
                }
                dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[0].name + "_" + str(random.randint(1,1000000)) + '.json'
                report_string = json.dumps(report)

                file = open(dst, 'w')
                file.write(report_string)
                file.close()

        generate_csv(path.replace("/", "_"), path, 1)


def generate_rescaled_maps(log_dir_name, paths):
    min_MinRadius = np.inf
    min_MeanLateralPosition = np.inf
    min_DirectionCoverage = np.inf
    min_SegmentCount = np.inf
    min_SDSteeringAngle = np.inf

    max_MinRadius = 0
    max_MeanLateralPosition = 0
    max_DirectionCoverage = 0
    max_SegmentCount = 0
    max_SDSteeringAngle = 0

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "results_MeanLateralPosition_MinRadius" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]
            

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "results_DirectionCoverage_MinRadius" in g]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_DirectionCoverage" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]

            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]



    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_MinRadius" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SegmentCount" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]

            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_DirectionCoverage" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]

            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_DirectionCoverage_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_DirectionCoverage > data["DirectionCoverage_min"]:
                min_DirectionCoverage = data["DirectionCoverage_min"]
            if max_DirectionCoverage < data["DirectionCoverage_max"]:
                max_DirectionCoverage = data["DirectionCoverage_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MinRadius_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MinRadius > data["MinRadius_min"]:
                min_MinRadius = data["MinRadius_min"]
            if max_MinRadius < data["MinRadius_max"]:
                max_MinRadius = data["MinRadius_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)

            if min_SegmentCount > data["SegmentCount_min"]:
                min_SegmentCount = data["SegmentCount_min"]
            if max_SegmentCount < data["SegmentCount_max"]:
                max_SegmentCount = data["SegmentCount_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SDSteeringAngle" in h]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            if min_MeanLateralPosition > data["MeanLateralPosition_min"]:
                min_MeanLateralPosition = data["MeanLateralPosition_min"]
            if max_MeanLateralPosition < data["MeanLateralPosition_max"]:
                max_MeanLateralPosition = data["MeanLateralPosition_max"]

            if min_SDSteeringAngle > data["SDSteeringAngle_min"]:
                min_SDSteeringAngle = data["SDSteeringAngle_min"]
            if max_SDSteeringAngle < data["SDSteeringAngle_max"]:
                max_SDSteeringAngle = data["SDSteeringAngle_max"]


    print("MinRadius: ", min_MinRadius, max_MinRadius)
    print("MeanLateralPosition: ", min_MeanLateralPosition, max_MeanLateralPosition)
    print("DirectionCoverage: ", min_DirectionCoverage, max_DirectionCoverage)
    print("SegmentCount: ", min_SegmentCount, max_SegmentCount)
    print("SDSteeringAngle: ", min_SDSteeringAngle, max_SDSteeringAngle)


    for path in paths:
        folders = sorted(glob.glob(path + "/*/"), key=os.path.getmtime)
        for folder in folders:
            jsons = [f for f in sorted(glob.glob(f"{folder}/*.json", recursive=True),key=os.path.getmtime) if "results_MeanLateralPosition_MinRadius" in f]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)

                    fts = list()

                    ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_MinRadius, max_MinRadius)
                    print("here")
                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()



            jsons = [g for g in sorted(glob.glob(f"{folder}/*.json", recursive=True),key=os.path.getmtime) if "results_DirectionCoverage_MinRadius" in g]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_DirectionCoverage, max_DirectionCoverage, min_MinRadius, max_MinRadius)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_DirectionCoverage" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_DirectionCoverage, max_DirectionCoverage)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()



            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_MinRadius" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    ft1 = FeatureDimension(name="MinRadius", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_SegmentCount, max_SegmentCount, min_MinRadius, max_MinRadius)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SegmentCount" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()



                    ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    ft1 = FeatureDimension(name="MeanLateralPosition", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SegmentCount, max_SegmentCount)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_DirectionCoverage" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    

                    ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    ft1 = FeatureDimension(name="SegmentCount", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_SegmentCount, max_SegmentCount, min_DirectionCoverage, max_DirectionCoverage)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_DirectionCoverage_SDSteeringAngle" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="DirectionCoverage", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_DirectionCoverage, max_DirectionCoverage, min_SDSteeringAngle, max_SDSteeringAngle)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_MinRadius_SDSteeringAngle" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="MinRadius", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_MinRadius, max_MinRadius, min_SDSteeringAngle, max_SDSteeringAngle)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_SegmentCount_SDSteeringAngle" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)

                    fts = list()

                    ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="SegmentCount", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_SegmentCount, max_SegmentCount, min_SDSteeringAngle, max_SDSteeringAngle)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()


            jsons = [h for h in sorted(glob.glob(f"{folder}/*.json", recursive=True), key=os.path.getmtime) if "results_MeanLateralPosition_SDSteeringAngle" in h]
            for json_data in jsons:
                with open(json_data) as json_file:
                    data = json.load(json_file)
                    fts = list()

                    ft1 = FeatureDimension(name="SDSteeringAngle", feature_simulator="min_radius", bins=1)
                    fts.append(ft1)

                    ft3 = FeatureDimension(name="MeanLateralPosition", feature_simulator="mean_lateral_position", bins=1)
                    fts.append(ft3)

                    performances = us.new_rescale(fts, np.array(data["Performances"]), min_MeanLateralPosition, max_MeanLateralPosition, min_SDSteeringAngle, max_SDSteeringAngle)

                    plot_heatmap(performances, fts[1],fts[0], savefig_path=path)

                    # filled values
                    total = np.size(performances)

                    filled = np.count_nonzero(performances != np.inf)
                    COUNT_MISS = 0

                    for (i, j), value in np.ndenumerate(performances):
                        if performances[i, j] != np.inf:
                                if performances[i, j] < 0:
                                    COUNT_MISS += 1

                    report = {
                        'Filled cells': str(filled),
                        'Filled density': str(filled / total),
                        'Misbehaviour': str(COUNT_MISS),
                        'Misbehaviour density': str(COUNT_MISS / filled)
                    }
                    dst = f"{path}/rescaled_" + fts[1].name + "_" + fts[
                        0].name + "_" + str(random.randint(1,1000000)) + '.json'
                    report_string = json.dumps(report)

                    file = open(dst, 'w')
                    file.write(report_string)
                    file.close()

        generate_csv(path.replace("/", "_"), path, 1)

                

def generate_csv(filename, log_dir_name, INTERVAL):
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    # write the header
    cf.writerow(["Features", "Time", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density"])

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_MeanLateralPosition_MinRadius" in f]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,MeanLateralPosition", id, data["Filled cells"], data["Filled density"],
                        data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "rescaled_DirectionCoverage_MinRadius" in g]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,DirectionCoverage", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_DirectionCoverage" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MeanLateralPosition,DirectionCoverage", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_MinRadius" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,SegmentCount", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SegmentCount" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MeanLateralPosition,SegmentCount", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_DirectionCoverage" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["DirectionCoverage,SegmentCount", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_DirectionCoverage_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["DirectionCoverage,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MinRadius_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_SegmentCount_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["SegmentCount,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "rescaled_MeanLateralPosition_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MeanLateralPosition,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL


def generate_csv_2(filename, log_dir_name, INTERVAL):
    filename = filename + ".csv"
    fw = open(filename, 'w')
    cf = csv.writer(fw, lineterminator='\n')

    # write the header
    cf.writerow(["Features", "Time", "Filled cells", "Filled density", 
                 "Misbehaviour", "Misbehaviour density"])

    jsons = [f for f in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "report_MeanLateralPosition_MinRadius" in f]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,MeanLateralPosition", id, data["Filled cells"], data["Filled density"],
                        data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [g for g in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True),key=os.path.getmtime) if "report_DirectionCoverage_MinRadius" in g]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,DirectionCoverage", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_DirectionCoverage" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MeanLateralPosition,DirectionCoverage", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_MinRadius" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,SegmentCount", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SegmentCount" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MeanLateralPosition,SegmentCount", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_DirectionCoverage" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["DirectionCoverage,SegmentCount", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_DirectionCoverage_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["DirectionCoverage,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MinRadius_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MinRadius,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_SegmentCount_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["SegmentCount,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL

    jsons = [h for h in sorted(glob.glob(f"{log_dir_name}/**/*.json", recursive=True), key=os.path.getmtime) if "report_MeanLateralPosition_SDSteeringAngle" in h]
    id = INTERVAL
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            cf.writerow(["MeanLateralPosition,SDSteeringAngle", id,  data["Filled cells"], data["Filled density"],
                          data["Misbehaviour"], data["Misbehaviour density"]])
            id += INTERVAL



if __name__ == "__main__":
    #generate_rescaled_maps("new-All-10", ["Initial"])
    generate_csv_2("initial_reports", "Initial_logs", 15)
