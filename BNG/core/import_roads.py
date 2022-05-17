import json
import glob
import sys
import os
import time
from pathlib import Path
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
from scipy.spatial import distance

from shapely.geometry import LineString, Point
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union

path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

import math
from core.mapelites_bng import MapElitesBNG
from self_driving.beamng_member import BeamNGMember
import self_driving.beamng_config as cfg
import self_driving.beamng_problem as BeamNGProblem
from self_driving.beamng_individual import BeamNGIndividual
import core.utils as us
from self_driving.beamng_road_imagery import BeamNGRoadImagery
from self_driving.simulation_data import SimulationDataRecordProperties
from self_driving.road_bbox import RoadBoundingBox
from self_driving.simulation_data import SimulationParams, SimulationDataRecords, SimulationData, SimulationDataRecord
from self_driving.vehicle_state_reader import VehicleState
from core.road_visualizer import RoadVisualizer
from core.road_profiler import RoadProfiler
from core.main_road_image import get_geometry
from self_driving.catmull_rom import catmull_rom

radius_threshold = 47

# method for generating map, it runs the simulation for each combination of features
def generate_maps(output_name, roads_path):
    roads = []
    jsons = [f for f in glob.glob(f"{roads_path}/**/*.json", recursive=True) if "simulation" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            control_nodes = data["road"]["nodes"]
            for node in control_nodes:
                node[2] = -28.0
            control_nodes = np.array(control_nodes)
            roads.append(control_nodes)

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    # The experiment folder
    if len(roads) > 0:
        for i in range(0, 10):
            map_E = MapElitesBNG(i, problem, "", True)
            for road in roads:
                bbox_size = (-250.0, 0.0, 250.0, 500.0)
                road_bbox = RoadBoundingBox(bbox_size)
                res = BeamNGMember([], [tuple(t) for t in road], len(road), road_bbox)
                res.config = config
                res.problem = problem
                individual: BeamNGIndividual = BeamNGIndividual(res, config)
                map_E.place_in_mapelites(individual)

            generate_results(map_E, output_name, False)


# method for generating map, it runs simulation only once
def generate_maps_onetime_simulation(output_name, roads_path):
    roads = []
    jsons = [f for f in glob.glob(f"{roads_path}/**/*.json", recursive=True) if "simulation.full" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            control_nodes = data["road"]["nodes"]
            for node in control_nodes:
                node[2] = -28.0
            control_nodes = np.array(control_nodes)
            roads.append(control_nodes)

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)

    map_E = MapElitesBNG(0, problem, "", True)
    individuals = []
    if len(roads) > 0:
        for road in roads:
            bbox_size = (-250.0, 0.0, 250.0, 500.0)
            road_bbox = RoadBoundingBox(bbox_size)
            res = BeamNGMember([], [tuple(t) for t in road], len(road), road_bbox)
            res.config = config
            res.problem = problem
            individual: BeamNGIndividual = BeamNGIndividual(res, config)
            individual.m.sample_nodes = us.new_resampling(individual.m.sample_nodes)
            map_E.performance_measure(individual)
            individuals.append(individual)

    for i in range(0, 10):
        map_E = MapElitesBNG(i, problem, "", True)
        for ind in individuals:
            map_E.place_in_mapelites_without_sim(ind)

        generate_results(map_E, output_name, False)        


# method for generating map without running simulations
def generate_maps_without_simulation(output_name, roads_path):
    # roads are 2D array with control_nodes and records
    roads = []
    jsons = [f for f in glob.glob(f"{roads_path}/**/*.json", recursive=True) if "simulation.full" in f]
    print(jsons)
    for json_data in jsons:
        if os.stat(json_data).st_size == 0:
            json_data = json_data + ".bkp"
        with open(json_data) as json_file:
            print(json_data)
            data = json.load(json_file)
            control_nodes = data["road"]["nodes"]
            for node in control_nodes:
                node[2] = -28.0
            control_nodes = np.array(control_nodes)
            records = data["records"]
            roads.append([control_nodes, records])

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    if len(roads) > 0:
        #for i in [4,6,9,10,11,12]:
        for i in [["SegmentCount", "MeanLateralPosition"],
        ["SegmentCount", "Curvature"],
        ["Curvature", "MeanLateralPosition"],
        ["SegmentCount", "SDSteeringAngle"],
        ["SDSteeringAngle", "MeanLateralPosition"],
        ["SDSteeringAngle", "Curvature"]]:
            map_E = MapElitesBNG(i, problem, "", True)
            for road in roads:
                bbox_size = (-250.0, 0.0, 250.0, 500.0)
                road_bbox = RoadBoundingBox(bbox_size)
                sample_nodes = us.new_resampling(road[0])
                member = BeamNGMember([], [tuple(t) for t in sample_nodes], len(road), road_bbox)
                member.config = config
                member.problem = problem
                simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
                simulation_data = SimulationData(sim_name)
                records = road[1]
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
                    map_E.place_in_mapelites_without_sim_and_count(individual)

            generate_results(map_E, output_name, False)



# method for generating map by putting sectors in cells
def generate_maps_by_sectors(output_name, roads_path):
    roads = []
    jsons = [f for f in sorted(glob.glob(f"{roads_path}/**/*.json", recursive=True), key=os.path.getmtime) if "simulation.full" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            print(json_data)
            data = json.load(json_file)
            control_nodes = data["road"]["nodes"]
            for node in control_nodes:
                node[2] = -28.0
            control_nodes = np.array(control_nodes)
            roads.append(control_nodes)

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    map_E = MapElitesBNG(["SegmentCount", "MeanLateralPosition"], problem, "", 1, True)
    individuals = []

    map_E = MapElitesBNG(0, problem, "",1, True)
    individuals = []
    i = 0
    if len(roads) > 0:
        for road in roads:
            middle, left, right = get_geometry(road)
            ### [{"right": [-443.02764892578125, 522.4307861328125, -9.765848517417908e-06], "left": [-436.89605712890625, 527.5692138671875, -9.765848517417908e-06], "middle": [-439.96185302734375, 525, -9.765848517417908e-06]},

            road_geometry = list()
            for index in range(len(middle)):
                point = dict()
                point['middle'] = middle[index]
                # Copy the Z value from middle
                point['right'] = list(right[index])
                point['right'].append(middle[index][2])
                # Copy the Z value from middle
                point['left'] = list(left[index])
                point['left'].append(middle[index][2])

                road_geometry.append( point )

            rv = RoadVisualizer(road_geometry, number_of_sectors=4)

            for sector in rv.sectors:
                control_nodes = list()
                # define a new individual for each sector
                sector_geometry = [element for rs in sector for element in rs.geometry]
                standard_sector_geometry = rv._standardize(sector_geometry)
                middle_edge_x = [e['middle'][0] for e in standard_sector_geometry]
                middle_edge_y = [e['middle'][1] for e in standard_sector_geometry]
                middle_edge_z = [e['middle'][2] for e in standard_sector_geometry]
                control_nodes.append([0.0, -20.0, -28.0, 8.0])
                control_nodes.append([0.0, -10.0, -28.0, 8.0])
                for point in zip(middle_edge_x, middle_edge_y, middle_edge_z):
                    point_list = list()
                    point_list.append(point[0])
                    point_list.append(point[1])
                    point_list.append(-28.0)
                    point_list.append(8.0)
                    if len(control_nodes) == 0 or point_list != control_nodes[-1]:
                        control_nodes.append(point_list)

                def _get_next_xy(x0: float, y0: float, angle: float):
                    angle_rad = math.radians(angle)
                    seg_length = 10
                    return x0 + seg_length * math.cos(angle_rad), y0 + seg_length * math.sin(angle_rad)

                for i in range(0, 3):
                    v = np.subtract(control_nodes[-1], control_nodes[-2])
                    start_angle = int(np.degrees(np.arctan2(v[1], v[0])))
                    angle = start_angle
                    x0, y0, z0, width0 = control_nodes[-1]
                    x1, y1 = _get_next_xy(x0, y0, angle)
                    control_nodes.append([x1, y1, -28.0, 8.0])

                bbox_size = (-500.0, 0.0, 500.0, 1000.0)
                road_bbox = RoadBoundingBox(bbox_size)
                num_spline_nodes = 10
                new_sample_nodes = us.new_resampling(control_nodes)
                res = BeamNGMember([tuple(t) for t in new_sample_nodes], [tuple(t) for t in new_sample_nodes],
                                   num_spline_nodes, road_bbox)
                
                res.problem = problem
                bbox_size = (-500.0, 0.0, 500.0, 1000.0)
                road_bbox = RoadBoundingBox(bbox_size)
                res.config = config
                res.problem = problem
                individual: BeamNGIndividual = BeamNGIndividual(res, config)
                map_E.performance_measure(individual)
                individuals.append(individual)

    for i in [["SegmentCount", "MeanLateralPosition"],
        ["SegmentCount", "Curvature"],
        ["Curvature", "MeanLateralPosition"],
        ["SegmentCount", "SDSteeringAngle"],
        ["SDSteeringAngle", "MeanLateralPosition"],
        ["SDSteeringAngle", "Curvature"]]:
        map_E = MapElitesBNG(i, problem, "", 1, True)
        for ind in individuals:
            map_E.place_in_mapelites_without_sim(ind)



def generate_maps_by_sectors_without_simulation(output_name, roads_path):
    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)

    individuals = []

    jsons = [f for f in sorted(glob.glob(f"{roads_path}/**/*.json", recursive=True), key=os.path.getmtime) if "simulation.full" in f]
    for json_data in jsons:
        if Path(json_data).stat().st_size == 0:
            json_data = json_data + ".bkp"
        with open(json_data) as json_file:
            print(json_data)
            simulation_data_dict = json.load(json_file)

            # Compare the first road node and the first position of the car to understand whether the car
            # drove in the opposite direction.
            road_nodes = simulation_data_dict["road"]["nodes"]
            road_nodes_with_tolerance = [ [r[0], r[1], r[2], r[3] + 4] for r in road_nodes ]

            # Initial road_node and final road node
            # start_of_the_road = Point(road_nodes[0][0], road_nodes[0][1])
            # end_of_the_road = Point(road_nodes[-1][0], road_nodes[-1][1])

            # Position of the first observation. In some cases, the initial states have position in the origin
            # which is wrong. We default to consider only records collected after 5.0 seconds of simulation.
            # TODO This is hardcoded and difficult to solve since we do not really now when the experiment
            #  really started

            # Take only the states observed after 1.0 seconds
            simulation_states = [s for s in simulation_data_dict["records"] if s["timer"] >= 1.0]
            valid_simulation_states = simulation_states
            if len(simulation_states) < 5:
                print("*    Sample %s is invalid, not enough states: %d." % (json_data,len(simulation_states)))
            else:

                first_sample = simulation_states[0]
                position_of_first_sample = Point(first_sample["pos"][0], first_sample["pos"][1])

                # Assuming that the first position is on the road, we assume it is correct, so we can identify the
                # right lane as the lane on which the car currently is !

                road_geometry = get_geometry2(road_nodes)
                road_geometry_with_tolerance = get_geometry2(road_nodes_with_tolerance)

                road_poly = _polygon_from_geometry(road_geometry)
                road_poly_with_tolerance = _polygon_from_geometry(road_geometry_with_tolerance)

                # TODO simplify using _polygon_from_geometry 'left', 'middle'
                left_edge_x = np.array([e['left'][0] for e in road_geometry])
                left_edge_y = np.array([e['left'][1] for e in road_geometry])
                middle_edge_x = np.array([e['middle'][0] for e in road_geometry])
                middle_edge_y = np.array([e['middle'][1] for e in road_geometry])
                right_edge_x = np.array([e['right'][0] for e in road_geometry])
                right_edge_y = np.array([e['right'][1] for e in road_geometry])

                # Left Lane
                right_edge = LineString(zip(middle_edge_x[::-1], middle_edge_y[::-1]))
                left_edge = LineString(zip(left_edge_x, left_edge_y))
                l_edge = left_edge.coords
                r_edge = right_edge.coords
                left_lane_polygon = Polygon(list(l_edge) + list(r_edge))

                # Right Lane
                right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
                left_edge = LineString(zip(middle_edge_x, middle_edge_y))
                l_edge = left_edge.coords
                r_edge = right_edge.coords
                right_lane_polygon = Polygon(list(l_edge) + list(r_edge))

                # Check where is the position_of_first_sample
                if right_lane_polygon.contains(position_of_first_sample):
                    reverse_road = False
                    print("Correct DIRECTION")
                elif left_lane_polygon.contains(position_of_first_sample):
                    reverse_road = True
                    print("Reversed DIRECTION")
                else:
                    raise Exception("Cannot establish the direction of the road")

                # Make sure that we read the roads node in the reversed order
                if reverse_road:
                    road_nodes.reverse()
                    road_nodes_with_tolerance.reverse()

                # Note that we reversed the nodes here !
                road_geometry = get_geometry2(road_nodes)
                road_geometry_with_tolerance = get_geometry2(road_nodes_with_tolerance)

                # Not sure this is required. Question is... how do I map the simulation data otherwise?
                # road_geometry = self.standardize(road_geometry)
                # Those values are meaninfull only if we compute sectors by travel time, not distance
                friction_coefficient = 0.8  # Friction Coefficient
                speed_limit_meter_per_sec = 90.0 / 3.6  # Speed limit m/s
                road_profiler = RoadProfiler(friction_coefficient, speed_limit_meter_per_sec, road_geometry)

                # TODO We use the wider road here, this should not change the definition of the sectors, since we
                #  should always use the middle points. ASSUME THAT THE SECTORS CORRESPOND ?!
                road_profiler_with_tolerance = RoadProfiler(friction_coefficient, speed_limit_meter_per_sec, road_geometry_with_tolerance)

                # This does not seem to work
                sectors = road_profiler.compute_sectors_by_travel_time(4)
                wider_sectors = road_profiler_with_tolerance.compute_sectors_by_travel_time(4)
                # sectors = road_profiler.compute_sectors_by_driving_distance(4)
                # wider_sectors = road_profiler_with_tolerance.compute_sectors_by_driving_distance(4)

                start_from = 0

                colors = ['black', 'green']
                for sector_idx, (sector, wider_sector) in enumerate(zip(sectors, wider_sectors)):
                    # Initial road point of the sector is the first point in the geometry of the first segment
                    initial_road_node = sector[0].geometry[0]['middle']
                    final_road_node = sector[-1].geometry[-1]['middle']

                    # Identify the portion of records/states that belong to this sector using the monitored position.
                    # sector_polygon = cascaded_union([segment.polygon for segment in sector])
                    # USE THE TOLERANCE
                    sector_polygon = cascaded_union([segment.polygon for segment in wider_sector])

                    initial_state = start_from
                    final_state = None

                    # In case of an OOB/OBE the simulation is interrupted, and no more data are available
                    # So we do not consider the other sectors. This ensures that we will not generate AsFault samples
                    # that ONLY have input features
                    if start_from >= len(simulation_states) - 1:
                        print(">> No more data, probably due to OOB.")
                        break

                    for state_idx, state in enumerate(simulation_states[start_from:], start=start_from):

                        point = Point(state["pos"][0], state["pos"][1])
                        if not sector_polygon.contains(point):
                            # At this point we identified the last state of this sector state_idx - 1
                            final_state = state_idx - 1

                            # It might happen that some sectors are without any observation, in that case we need to
                            # reject them.
                            if final_state < initial_state:
                                # print(">> EMPTY Sector. Skip it")
                                break

                            # Generate the sample
                            # print("Generating a sample using the states from ", initial_state, "to", final_state)
                            if final_state - initial_state < 30:
                                print("*    Not enough states (%d) to make a sample from %s" %( (final_state - initial_state), json_data))
                            else:
                                start_sector_road = road_nodes.index(initial_road_node)
                                end_sector_road = road_nodes.index(final_road_node)

                                _road_nodes = road_nodes[start_sector_road:end_sector_road+1]

                                # Limit the simulation data to the sector between startd_idx and final_idx. The final idx must be included
                                # in the slice
                                _simulation_states = valid_simulation_states[initial_state:final_state+1]
                                bbox_size = (-250.0, 0.0, 250.0, 500.0)
                                road_bbox = RoadBoundingBox(bbox_size)
                                sample_nodes = us.new_resampling(_road_nodes)
                                member = BeamNGMember([], [tuple(t) for t in sample_nodes], 20, road_bbox)
                                member.config = config
                                member.problem = problem
                                simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                                sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
                                simulation_data = SimulationData(sim_name)
                                states = []
                                for record in _simulation_states:
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
                                    individuals.append(individual)
                            # Reset the index and restart
                            start_from = state_idx
                            break
                        else:
                            pass
                            # print("Adding state", state_idx, "as point", point.x, point.y, "to sector", sector_idx)
                            # plt.plot(point.x, point.y, 'o', color=colors[sector_idx % 2], alpha=0.2, markersize=12)

                    # We get here if there are no more states to process. This is either because the road is longer or
                    # there was an OOB.
                    if final_state is None:
                        # DEBUG
                        # print(">> Generating a sample using 'incomplete' data, possibly due to an OOB "
                        #       "for", simulation_full_path, "from:", initial_state, "to:", final_state)
                        final_state = len(simulation_states) - 1

                        if final_state - initial_state < 30:
                            print("*    Not enough states (%d) to make a sample from %s" %( (final_state - initial_state), json_data))
                        else:
                            start_sector_road = road_nodes.index(initial_road_node)
                            end_sector_road = road_nodes.index(final_road_node)

                            _road_nodes = road_nodes[start_sector_road:end_sector_road+1]

                            # Limit the simulation data to the sector between startd_idx and final_idx. The final idx must be included
                            # in the slice
                            _simulation_states = valid_simulation_states[initial_state:final_state+1]
                            bbox_size = (-250.0, 0.0, 250.0, 500.0)
                            road_bbox = RoadBoundingBox(bbox_size)
                            sample_nodes = us.new_resampling(_road_nodes)
                            member = BeamNGMember([], [tuple(t) for t in sample_nodes], 20, road_bbox)
                            member.config = config
                            member.problem = problem
                            simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
                            sim_name = member.config.simulation_name.replace('$(id)', simulation_id)
                            simulation_data = SimulationData(sim_name)
                            states = []
                            for record in _simulation_states:
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
                                individuals.append(individual)

                        # Make sure we update the indices
                        start_from = state_idx
     
    for i in [ ["SegmentCount", "MeanLateralPosition"],  ["Curvature", "MeanLateralPosition"],  ["SDSteeringAngle", "MeanLateralPosition"],
     ["SegmentCount", "Curvature"],  ["SegmentCount", "SDSteeringAngle"],  ["SDSteeringAngle", "Curvature"]]:
        map_E = MapElitesBNG(i, problem, "", 1, True)
        for individual in individuals:
            map_E.place_in_mapelites_without_sim(individual)
        generate_results(map_E, output_name, False)


def test_features_for_asFault(output_name, roads_path):
    roads = []
    jsons = [f for f in glob.glob(f"{roads_path}/**/*.json", recursive=True) if "simulation.full" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            control_nodes = data["road"]["nodes"]
            for node in control_nodes:
                node[2] = -28.0
            control_nodes = np.array(control_nodes)
            roads.append(control_nodes)

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)

    if len(roads) > 0:
        for road in roads:
            middle, left, right = get_geometry(road)
            ### [{"right": [-443.02764892578125, 522.4307861328125, -9.765848517417908e-06], "left": [-436.89605712890625, 527.5692138671875, -9.765848517417908e-06], "middle": [-439.96185302734375, 525, -9.765848517417908e-06]},

            road_geometry = list()
            for index in range(len(middle)):
                point = dict()
                point['middle'] = middle[index]
                # Copy the Z value from middle
                point['right'] = list(right[index])
                point['right'].append(middle[index][2])
                # Copy the Z value from middle
                point['left'] = list(left[index])
                point['left'].append(middle[index][2])

                road_geometry.append( point )

            rv = RoadVisualizer(road_geometry, number_of_sectors=4)

            for sector in rv.sectors:
                sample_nodes = list()
                # define a new individual for each sector
                sector_geometry = [element for rs in sector for element in rs.geometry]
                standard_sector_geometry = rv._standardize(sector_geometry)
                middle_edge_x = [e['middle'][0] for e in standard_sector_geometry]
                middle_edge_y = [e['middle'][1] for e in standard_sector_geometry]
                middle_edge_z = [e['middle'][2] for e in standard_sector_geometry]
                for point in zip(middle_edge_x, middle_edge_y, middle_edge_z):
                    point_list = []
                    point_list.append(point[0])
                    point_list.append(point[1])
                    point_list.append(point[2])
                    point_list.append(8.0)
                    if len(sample_nodes) == 0 or point_list != sample_nodes[-1]:
                        sample_nodes.append(point_list)

                new_sample_nodes = us.new_resampling(sample_nodes)
                bbox_size = (-500.0, 0.0, 500.0, 1000.0)
                road_bbox = RoadBoundingBox(bbox_size)
                res = BeamNGMember([], [tuple(t) for t in new_sample_nodes], len(new_sample_nodes), road_bbox)
                res.config = config
                res.problem = problem
                individual: BeamNGIndividual = BeamNGIndividual(res, config)
                count, segments = us.segment_count(individual)

                points_xx = []
                points_yy = []
                for point in individual.m.sample_nodes:
                    points_xx.append(point[0])
                    points_yy.append(point[1])
                [plt.plot(m, n, marker='.', color='black') for m,n in zip(points_xx, points_yy)]
                
                i = 0
                for segment in segments:
                    points, angles = map(list,zip(*segment))
                    # print(f"average angle: {np.mean(angles)}")
                    # print(f"std angle: {np.std(angles)}")
                    # print(f"number of points: {len(points)}")
                    
               


                    points_x = []
                    points_y = []
                    for point in points:
                        points_x.append(point[0])
                        points_y.append(point[1])
                    if i % 2 == 0:
                        [plt.plot(m, n, marker='.', color='red') for m,n in zip(points_x, points_y)]
                    else:
                        [plt.plot(m, n, marker='.', color='blue') for m,n in zip(points_x, points_y)]

                    i += 1
                

                if (i-1)%2 != 0:
                    points_xxx = []
                    points_yyy = []
                    j = 0
                    for i in range(len(points_xx)):
                        if points_x[-1] == points_xx[i]:
                            j = i
                            break
                    for jj in range(j, len(points_xx)):
                        points_xxx.append(points_xx[jj])
                        points_yyy.append(points_yy[jj])
                    [plt.plot(m, n, marker='.', color='blue') for m,n in zip(points_xxx, points_yyy)]

                    
                plt.axes().set_aspect('equal', 'datalim')            
                plt.show()


def test_features_for_DeepJanus(outputname, roads_path):
    roads = []
    jsons = [f for f in glob.glob(f"{roads_path}/**/*.json", recursive=True) if "simulation.full" in f]
    for json_data in jsons:
        with open(json_data) as json_file:
            data = json.load(json_file)
            control_nodes = data["road"]["nodes"]
            for node in control_nodes:
                node[2] = -28.0
            control_nodes = np.array(control_nodes)
            roads.append(control_nodes)

    config = cfg.BeamNGConfig()
    problem = BeamNGProblem.BeamNGProblem(config)
    if len(roads) > 0:
        for road in roads:
            bbox_size = (-250.0, 0.0, 250.0, 500.0)
            road_bbox = RoadBoundingBox(bbox_size)
            res = BeamNGMember([], [tuple(t) for t in road], len(road), road_bbox)
            res.config = config
            res.problem = problem

            r= 0.2
            n = 20
            t = np.linspace(0, 2*np.pi, n)
            x = r * np.cos(t)
            y = r * np.sin(t)

            points = []
            for x, y in zip(x, y):
                point = tuple([x, y])
                points.append(point)


            individual: BeamNGIndividual = BeamNGIndividual(res, config)
            individual.m.sample_nodes = points
            individual.m.sample_nodes = us.new_resampling(individual.m.sample_nodes)
            count, segments = us.segment_count(individual)
            
            print(f"segment_count:{count}")
            points_xx = []
            points_yy = []
            for point in individual.m.sample_nodes:
                points_xx.append(point[0])
                points_yy.append(point[1])
            [plt.plot(m, n, marker='.', color='black') for m,n in zip(points_xx, points_yy)]
            
            i = 0
            for segment in segments:
                points, angles = map(list,zip(*segment))
                # print(f"average angle: {np.mean(angles)}")
                # print(f"std angle: {np.std(angles)}")
                # print(f"number of points: {len(points)}")
                # points = segment
                

                
                points_x = []
                points_y = []
                for point in points:
                    points_x.append(point[0])
                    points_y.append(point[1])
                if i % 2 == 0:
                    [plt.plot(m, n, marker='.', color='red') for m,n in zip(points_x, points_y)]
                else:
                    [plt.plot(m, n, marker='.', color='blue') for m,n in zip(points_x, points_y)]

                i += 1
            

            if (i-1)%2 != 0:
                points_xxx = []
                points_yyy = []
                j = 0
                for i in range(len(points_xx)):
                    if points_x[-1] == points_xx[i]:
                        j = i
                        break
                for jj in range(j, len(points_xx)):
                    points_xxx.append(points_xx[jj])
                    points_yyy.append(points_yy[jj])
                [plt.plot(m, n, marker='.', color='blue') for m,n in zip(points_xxx, points_yyy)]

            print("###############################################")
            # for point in point_angles:
            #     points_xx.append(point[0])
            #     points_yy.append(point[1])
            # [plt.plot(m, n, marker='.', color='black') for m,n in zip(points_xx, points_yy)]

            # colors = ['red', 'blue', 'green', 'cyan', 'magenta']
            # # import matplotlib.pyplot as plt
            # j = 0
            # for i in [3, 5, 7, 9, 11]:
            #     radius, curve = us.new_min_radius(individual, i)
            #     radius = int(radius)
            #     points_x = []
            #     points_y = []
            #     for point in curve:
            #         points_x.append(point[0])
            #         points_y.append(point[1])

            #     [plt.plot(m, n, marker='o', color=colors[j]) for m,n in zip(points_x, points_y)]
            #     j += 1
            
            #     if i == 3:
            #         red_patch = mpatches.Patch(color='red', label=f'min_radius:{radius}')
            #     elif i == 5:
            #         blue_patch = mpatches.Patch(color='blue', label=f'min_radius(3):{radius}')
            #     elif i == 7:
            #         green_patch = mpatches.Patch(color='green', label=f'min_radius(5):{radius}')
            #     elif i == 9:
            #         cyan_patch = mpatches.Patch(color='cyan', label=f'min_radius(7):{radius}')
            #     elif i == 11:
            #         magenta_patch = mpatches.Patch(color='magenta', label=f'min_radius(10):{radius}')
                    
            # plt.legend(handles=[red_patch, blue_patch, green_patch, cyan_patch, magenta_patch])  
            
            plt.axes().set_aspect('equal', 'datalim')        
            plt.show()


# method for reporting the results
def generate_results(map_E, output_name, rescaled):
    # The experiment folder
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir_name = f"log_{now}"
    if rescaled:
        log_dir_name = log_dir_name + "_rescaled"

    map_E.log_dir_path = log_dir_name
    log_dir_path = Path(f"{output_name}_logs/{log_dir_name}")
    log_dir_path.mkdir(parents=True, exist_ok=True)

    image_dir = f'{output_name}_logs/{log_dir_name}/{map_E.feature_dimensions[1].name}_{map_E.feature_dimensions[0].name}'

    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    # filled values
    filled = np.count_nonzero(map_E.solutions != None)
    total = np.size(map_E.solutions)
    filled_density = (filled / total)

    COUNT_MISS = 0
    covered_seeds = set()
    mis_seeds = set()
    for (i, j), value in np.ndenumerate(map_E.performances):
        if map_E.performances[i, j] != np.inf:
            covered_seeds.add(map_E.solutions[i, j].seed)
            if map_E.performances[i, j] < 0:
                mis_seeds.add(map_E.solutions[i, j].seed)
                COUNT_MISS += 1

            destination_sim = f"{image_dir_path}\\simulation_{map_E.solutions[i, j].m.name}_{i, j}.tsv"
            destination_sim_json = f"{image_dir_path}\\simulation_{map_E.solutions[i, j].m.name}_{i, j}.json"
            destination_road = f"{image_dir_path}/road_{map_E.solutions[i, j].m.name}_{i, j}.json"

            # with open(destination_sim_json, 'w') as f:
            #     f.write(json.dumps({
            #         map_E.solutions[i, j].m.simulation.f_params: map_E.solutions[i, j].m.simulation.params._asdict(),
            #         map_E.solutions[i, j].m.simulation.f_info: map_E.solutions[i, j].m.simulation.info.__dict__,
            #         map_E.solutions[i, j].m.simulation.f_road: map_E.solutions[i, j].m.simulation.road.to_dict(),
            #         map_E.solutions[i, j].m.simulation.f_records: [r._asdict() for r in
            #                                                        map_E.solutions[i, j].m.simulation.states]
            #     }))

            # with open(destination_sim, 'w') as f:
            #     sep = '\t'
            #     f.write(sep.join(SimulationDataRecordProperties) + '\n')
            #     gen = (r._asdict() for r in map_E.solutions[i, j].m.simulation.states)
            #     gen2 = (sep.join([str(d[key]) for key in SimulationDataRecordProperties]) + '\n' for d in gen)
            #     f.writelines(gen2)

            with open(destination_road, 'w') as f:
                f.write(json.dumps(map_E.solutions[i, j].m.to_dict()))

            road_imagery = BeamNGRoadImagery.from_sample_nodes(map_E.solutions[i, j].m.sample_nodes)
            image_path = image_dir_path.joinpath(f"img_{map_E.solutions[i, j].m.name}_{i, j}")
            road_imagery.save(image_path.with_suffix('.jpg'))
            #road_imagery.save(image_path.with_suffix('.svg'))

    print("generate results")
    filled_dists = []
    filled2 = []
    missed = []
    missed_dists = []
    
    for (i, j), value in np.ndenumerate(map_E.solutions):
        if value != None:
            filled2.append((i,j))
        if map_E.performances[i, j] != np.inf:
            if map_E.performances[i, j] < 0:
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
        'Covered seeds': len(covered_seeds),
        'Filled cells': str(filled),
        'Filled density': str(filled_density),
        'Misbehaviour seeds': len(mis_seeds),
        'Misbehaviour': str(COUNT_MISS),
        'Misbehaviour density': str(COUNT_MISS / filled),
        'Misbehaviour Sparsness': str(missed_sp),
        'Filled Sparsness': str(filled_sp),

    }
    dst = f"{name}_logs/{log_dir_name}/report_" + map_E.feature_dimensions[1].name + '_' + \
          map_E.feature_dimensions[0].name + '_' + str(now) + '.json'
    report_string = json.dumps(report)

    file = open(dst, 'w')
    file.write(report_string)
    file.close()

    map_E.plot_map_of_elites(map_E.performances, f"{name}_logs/{log_dir_name}")

    repo = {
        f"{map_E.feature_dimensions[1].name}_min": map_E.feature_dimensions[1].min,
        f"{map_E.feature_dimensions[0].name}_min": map_E.feature_dimensions[0].min,
        f"{map_E.feature_dimensions[1].name}_max": map_E.feature_dimensions[1].bins,
        f"{map_E.feature_dimensions[0].name}_max": map_E.feature_dimensions[0].bins,
        "Performances": map_E.performances.tolist(),
        #"Archive": map_E.archive.tolist()
    }
    filename = f"{name}_logs/{log_dir_name}/results_{map_E.feature_dimensions[1].name}_{map_E.feature_dimensions[0].name}.json"
    with open(filename, 'w') as f:
        f.write(json.dumps(repo))

def get_max_distance_from_set(ind, solution):
    distances = list()
    # print("ind:", ind)
    # print("solution:", solution)
    ind_spine = ind[0]

    for road in solution:
        road_spine = road[0]
        distances.append(manhattan_dist(ind_spine, road_spine))
    distances.sort()
    return distances[-1]

def manhattan_dist(ind1, ind2):
    return distance.cityblock(ind1, ind2)


def calc_point_edges(p1, p2):
    origin = np.array(p1[0:2])

    a = np.subtract(p2[0:2], origin)
    # print(p1, p2)
    v = (a / np.linalg.norm(a)) * p1[3] / 2

    l = origin + np.array([-v[1], v[0]])
    r = origin + np.array([v[1], -v[0]])
    return tuple(l), tuple(r)

# def get_geometry(middle_nodes):
#     middle = []
#     right = []
#     left = []
#     n = len(middle) + len(middle_nodes)

#     middle += list(middle_nodes)
#     left += [None] * len(middle_nodes)
#     right += [None] * len(middle_nodes)
#     for i in range(n - 1):
#         l, r = calc_point_edges(middle[i], middle[i + 1])
#         left[i] = l
#         right[i] = r
#     # the last middle point
#     right[-1], left[-1] = calc_point_edges(middle[-1], middle[-2])

#     road_geometry = list()
#     for index in range(len(middle)):
#         point = dict()
#         point['middle'] = middle[index]
#         # Copy the Z value from middle
#         point['right'] = list(right[index])
#         point['right'].append(middle[index][2])
#         # Copy the Z value from middle
#         point['left'] = list(left[index])
#         point['left'].append(middle[index][2])

#         road_geometry.append(point)

#     return road_geometry

def _polygon_from_geometry(road_geometry, left_side='left', right_side='right'):
    left_edge_x = np.array([e[left_side][0] for e in road_geometry])
    left_edge_y = np.array([e[left_side][1] for e in road_geometry])
    right_edge_x = np.array([e[right_side][0] for e in road_geometry])
    right_edge_y = np.array([e[right_side][1] for e in road_geometry])

    right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    left_edge = LineString(zip(left_edge_x, left_edge_y))
    l_edge = left_edge.coords
    r_edge = right_edge.coords

    return Polygon(list(l_edge) + list(r_edge))

def split_and_generate(roads_path):
    """
        Create a number of instances of AsFaultSample computed using the "self.n_sectors" from a single AsFault
        esperiment
    Args:
        simulation_full_path:

    Returns:

    """
    j = 0
    jsons = [f for f in sorted(glob.glob(f"{roads_path}/**/*.json", recursive=True), key=os.path.getmtime) if "simulation.full" in f]
    for json_data in jsons:
        j += 1
        if j < 11:
            print("**********")
            road_name = f"road{j}"
            with open(json_data) as json_file:
                simulation_data_dict = json.load(json_file)

                # Compare the first road node and the first position of the car to understand whether the car
                # drove in the opposite direction.
                road_nodes = simulation_data_dict["road"]["nodes"]
                road_nodes_with_tolerance = [ [r[0], r[1], r[2], r[3] + 4] for r in road_nodes ]

                # Initial road_node and final road node
                # start_of_the_road = Point(road_nodes[0][0], road_nodes[0][1])
                # end_of_the_road = Point(road_nodes[-1][0], road_nodes[-1][1])

                # Position of the first observation. In some cases, the initial states have position in the origin
                # which is wrong. We default to consider only records collected after 5.0 seconds of simulation.
                # TODO This is hardcoded and difficult to solve since we do not really now when the experiment
                #  really started

                # Take only the states observed after 5.0 seconds
                simulation_states = [s for s in simulation_data_dict["records"] if s["timer"] >= 5.0]
                first_sample = simulation_states[0]
                position_of_first_sample = Point(first_sample["pos"][0], first_sample["pos"][1])

                # Assuming that the first position is on the road, we assume it is correct, so we can identify the
                # right lane as the lane on which the car currently is !

                road_geometry = get_geometry(road_nodes)
                road_geometry_with_tolerance = get_geometry(road_nodes_with_tolerance)

                road_poly = _polygon_from_geometry(road_geometry)
                road_poly_with_tolerance = _polygon_from_geometry(road_geometry_with_tolerance)

                # TODO simplify using _polygon_from_geometry 'left', 'middle'
                left_edge_x = np.array([e['left'][0] for e in road_geometry])
                left_edge_y = np.array([e['left'][1] for e in road_geometry])
                middle_edge_x = np.array([e['middle'][0] for e in road_geometry])
                middle_edge_y = np.array([e['middle'][1] for e in road_geometry])
                right_edge_x = np.array([e['right'][0] for e in road_geometry])
                right_edge_y = np.array([e['right'][1] for e in road_geometry])

                # Left Lane
                right_edge = LineString(zip(middle_edge_x[::-1], middle_edge_y[::-1]))
                left_edge = LineString(zip(left_edge_x, left_edge_y))
                l_edge = left_edge.coords
                r_edge = right_edge.coords
                left_lane_polygon = Polygon(list(l_edge) + list(r_edge))

                # Right Lane
                right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
                left_edge = LineString(zip(middle_edge_x, middle_edge_y))
                l_edge = left_edge.coords
                r_edge = right_edge.coords
                right_lane_polygon = Polygon(list(l_edge) + list(r_edge))

                # Check where is the position_of_first_sample
                if right_lane_polygon.contains(position_of_first_sample):
                    reverse_road = False
                    #print("Correct DIRECTION")
                elif left_lane_polygon.contains(position_of_first_sample):
                    reverse_road = True
                    #print("Reversed DIRECTION")
                else:
                    raise Exception("Cannot establish the direction of the road")




                # Make sure that we read the roads node in the reversed order
                if reverse_road:
                    road_nodes.reverse()
                    road_nodes_with_tolerance.reverse()

                # Note that we reversed the nodes here !
                road_geometry = get_geometry(road_nodes)
                road_geometry_with_tolerance = get_geometry(road_nodes_with_tolerance)

                # Not sure this is required. Question is... how do I map the simulation data otherwise?
                # road_geometry = self.standardize(road_geometry)
                # Those values are meaninfull only if we compute sectors by travel time, not distance
                friction_coefficient = 0.8  # Friction Coefficient
                speed_limit_meter_per_sec = 90.0 / 3.6  # Speed limit m/s
                road_profiler = RoadProfiler(friction_coefficient, speed_limit_meter_per_sec, road_geometry)

                # TODO We use the wider road here, this should not change the definition of the sectors, since we
                #  should always use the middle points. ASSUME THAT THE SECTORS CORRESPOND ?!
                road_profiler_with_tolerance = RoadProfiler(friction_coefficient, speed_limit_meter_per_sec, road_geometry_with_tolerance)

                # This does not seem to work
                # sectors = road_profiler.compute_sectors_by_travel_time(self.n_sectors)
                sectors = road_profiler.compute_sectors_by_driving_distance(6)
                wider_sectors = road_profiler_with_tolerance.compute_sectors_by_driving_distance(6)

                start_from = 0

                colors = ['black', 'green']
                k = 0
                for sector_idx, (sector, wider_sector) in enumerate(zip(sectors, wider_sectors)):
                    
                    # Initial road point of the sector is the first point in the geometry of the first segment
                    initial_road_node = sector[0].geometry[0]['middle']
                    final_road_node = sector[-1].geometry[-1]['middle']

                    # Identify the portion of records/states that belong to this sector using the monitored position.
                    # sector_polygon = cascaded_union([segment.polygon for segment in sector])
                    # USE THE TOLERANCE
                    sector_polygon = cascaded_union([segment.polygon for segment in wider_sector])

                    initial_state = start_from
                    final_state = None

                    # In case of an OOB/OBE the simulation is interrupted, and no more data are available
                    # So we do not consider the other sectors. This ensures that we will not generate AsFault samples
                    # that ONLY have input features
                    if start_from >= len(simulation_states) - 1:
                        # print(">> No more data, probably due to OOB.")
                        break

                    for state_idx, state in enumerate(simulation_states[start_from:], start=start_from):

                        point = Point(state["pos"][0], state["pos"][1])

                        if not sector_polygon.contains(point):
                            k += 1
                            sector_name = f"{road_name}_{k}"
                            # At this point we identified the last state of this sector state_idx - 1
                            final_state = state_idx - 1

                            # It might happen that some sectors are without any observation, in that case we need to
                            # reject them.
                            if final_state < initial_state:
                                # print(">> EMPTY Sector. Skip it")
                                break

                            # Generate the sample
                            # print("Generating a sample using the states from ", initial_state, "to", final_state)
                            # samples.append(AsFaultSample(simulation_full_path, sector_idx,
                            #                              initial_road_node, final_road_node,
                            #                              initial_state, final_state, reversed=reverse_road))
                            
                            # If the road was driven in the opposite direction
                            nnodes = road_nodes 
                            # if reversed:
                            #     nnodes.reverse()
                          
                            start_sector_road = nnodes.index(initial_road_node)
                            end_sector_road = nnodes.index(final_road_node)

                            

                            nnodes = nnodes[start_sector_road:end_sector_road+1]

                            

                            plt.cla()
                            plt.clf()
                            points_xx = []
                            points_yy = []
                            nnodes = us.new_resampling(nnodes)
                            print(len(nnodes))
                            for point in nnodes:    
                                points_xx.append(point[0]) 
                                points_yy.append(point[1])
                            [plt.plot(m, n, marker='.', color='black') for m, n in zip(points_xx, points_yy)]
                            plt.axes().set_aspect('equal', 'datalim')
                            plt.savefig(f"{sector_name}.png")
                            plt.close()

                            # Reset the index and restart
                            start_from = state_idx
                            break
                        else:
                            pass
                            # print("Adding state", state_idx, "as point", point.x, point.y, "to sector", sector_idx)
                            # plt.plot(point.x, point.y, 'o', color=colors[sector_idx % 2], alpha=0.2, markersize=12)

                    # We get here if there are no more states to process. This is either because the road is longer or
                    # there was an OOB.
                    if final_state is None:
                        k += 1
                        sector_name = f"{road_name}_{k}"
                        # DEBUG
                        # print(">> Generating a sample using 'incomplete' data, possibly due to an OOB "
                        #       "for", simulation_full_path, "from:", initial_state, "to:", final_state)
                        final_state = len(simulation_states) - 1

                        # samples.append(AsFaultSample(simulation_full_path, sector_idx,
                        #                              initial_road_node, final_road_node,
                        #                              initial_state, final_state,
                        #                              reversed=reverse_road))
                        nnodes = road_nodes 
                        # if reversed:
                        #     nnodes.reverse()
                            
                                                  
                        start_sector_road = nnodes.index(initial_road_node)
                        end_sector_road = nnodes.index(final_road_node)


                        nnodes = nnodes[start_sector_road:end_sector_road+1]
                        
                        plt.cla()
                        plt.clf()
                        points_xx = []
                        points_yy = []
                        nnodes = us.new_resampling(nnodes)
                        print(len(nnodes))
                        for point in nnodes:    
                            points_xx.append(point[0]) 
                            points_yy.append(point[1])
                        [plt.plot(m, n, marker='.', color='black') for m, n in zip(points_xx, points_yy)]
                        plt.axes().set_aspect('equal', 'datalim')
                        plt.savefig(f"{sector_name}.png")
                        plt.close()


                        # Make sure we update the indices
                        start_from = state_idx

                        # DEBUG - PLOT

                        #
                        # debug_x = [state["pos"][0] for state in simulation_data_dict["records"][initial_state:final_state+1]]
                        # debug_y = [state["pos"][1] for state in simulation_data_dict["records"][initial_state:final_state+1]]
                        #
                        # map_size = 500
                        # plt.gca().set(xlim=(-30 -map_size, map_size + 30), ylim=(-30 - map_size, map_size + 30))
                        # plt.gca().set_aspect('equal', 'box')
                        #
                        # map_patch = patches.Rectangle((-map_size, -map_size), 2 * map_size, 2 * map_size, linewidth=1, edgecolor='black',
                        #                               facecolor='none')
                        # plt.gca().add_patch(map_patch)
                        #
                        # map_size = 450
                        # map_patch = patches.Rectangle((-map_size, -map_size), 2 * map_size, 2 * map_size, linewidth=1,
                        #                               edgecolor='gray', facecolor='none')
                        #
                        # plt.gca().add_patch(map_patch)
                        # plt.plot(*sector_polygon.exterior.xy, color=colors[sector_idx % 2])

                        # plt.plot(*right_lane_polygon.exterior.xy, color="black", alpha=0.5)

                        # Plot the map. Trying to re-use an artist in more than one Axes which is supported
                        # plt.plot(debug_x, debug_y, 'o', color=colors[sector_idx % 2], alpha=0.2, markersize=12)
                        # plt.plot(debug_x[-1], debug_y[-1], 'o', color=colors[sector_idx % 2], alpha=1, markersize=12)
  

# main
name = str(sys.argv[1])
path = sys.argv[2]
simulation = int(sys.argv[3])

if simulation == 0:
    generate_maps(name, path)
elif simulation == 1:
    generate_maps_onetime_simulation(name, path)
elif simulation == 2:
    generate_maps_without_simulation(name, path)
elif simulation == 3:
    generate_maps_by_sectors(name, path)
elif simulation == 4:
    test_features_for_asFault(name, path)
elif simulation == 5:
    split_and_generate(path)
elif simulation == 6:
    generate_maps_by_sectors_without_simulation(name, path)
else:
    test_features_for_DeepJanus(name, path)
