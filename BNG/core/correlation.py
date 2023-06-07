import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

#sys.path.insert(0, r'C:\DeepHyperion-BNG')
#sys.path.append(os.path.dirname(os.path.dirname(os.path.join(__file__))))
path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))

import numpy as np
import re
import core.utils as us
import csv
import json
from scipy import stats
from main_road_image import get_geometry
from db_utils import DBUtils
from road_visualizer import RoadVisualizer
import road_profiler
from self_driving.beamng_member import BeamNGMember
import self_driving.beamng_config as cfg
import self_driving.beamng_problem as BeamNGProblem
from self_driving.beamng_individual import BeamNGIndividual
from self_driving.road_bbox import RoadBoundingBox

# tags
smoothness= []
complexity = []
orientation = []

list_of_tag = []
list_of_road = []

# asfault individuals
asfault_roads = []
# deepjanus individuals
deepjanus_roads = []

asfault_input_file = sys.argv[1]   
deepjanus_input_file = sys.argv[2] 
asfault_db = sys.argv[3]
deepjanus_db = sys.argv[4]

# total roads
road_ids = []

# metrics
minRadius = []
curvatureProfile = []    
directionCoverage = []
segmentCount = []
relSegment = []
segmentCountWithoutStraight = []
relSegmentWithoutStraight= []
segmentScore = []


config = cfg.BeamNGConfig()
problem = BeamNGProblem.BeamNGProblem(config)


################################### AsFault ##########################

# Input must be a CSV file that contains the id of the roads to plot. Roads are stored inside the db
with open(asfault_input_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # some refactoring added to rows for simplicity!      
        row = str(row).replace(";t", ",t").split(";")
        road = row[1].replace('"', '').replace('json.interactive_plot.json', 'json')
        list_of_road.append(road)            
        list_of_tag.append((road,row[3].replace('"', '').replace('\'', '')))
        

database_asfault = DBUtils(asfault_db)
roads_data_asfault = database_asfault.get_roads(list_of_road)


# sorting roads as they are in csv of tags
res = [next((sub for sub in roads_data_asfault  
       if sub[0] == i), 0) for i in list_of_road] 


# distinguish sectors of roads and create a new individual for each
for road_id, road_spine in res:
    roads = json.loads(road_spine)
    middle, left, right = get_geometry(roads)
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
    #rv._plot_sectors(road_id)

    sector_num = 0
    for sector in rv.sectors:
        road_ids.append(road_id+ "_" +str(sector_num))
        sample_nodes = list()
        # define a new individual for each sector
        sector_geometry = [element for rs in sector for element in rs.geometry]
        middle_edge_x = [e['middle'][0] for e in sector_geometry]
        middle_edge_y = [e['middle'][1] for e in sector_geometry]
        middle_edge_z = [e['middle'][2] for e in sector_geometry]
        
        for point in zip(middle_edge_x, middle_edge_y, middle_edge_z):
            point_list = []
            point_list.append(point[0])
            point_list.append(point[1])
            point_list.append(point[2])
            sample_nodes.append(point_list)
        
        bbox_size = (-250.0, 0.0, 250.0, 500.0)
        road_bbox = RoadBoundingBox(bbox_size)
        res = BeamNGMember([], [tuple(t) for t in sample_nodes], len(sample_nodes), road_bbox)
        res.config = config
        res.problem = problem
        individual: BeamNGIndividual = BeamNGIndividual(res, config)
        individual.m.sample_nodes = us.new_resampling(individual.m.sample_nodes)
        asfault_roads.append(individual)
        sector_num += 1

# extract tags for each sector of each as fault road
for road in list_of_road:
    for _tag in list_of_tag:
        if _tag[0] == road:
            tag = _tag[1]

    # 3 or 4 sectors
    pattern = re.compile('|'.join(['tag\-for\-sector\-0=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]],tag\-for\-sector\-1=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]],tag\-for\-sector\-2=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]],tag\-for\-sector\-3=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]',
    'tag\-for\-sector\-0=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]],tag\-for\-sector\-1=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]],tag\-for\-sector\-2=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]']))

    sectors = pattern.findall(tag)  
    for sector in sectors:
        # road with 4 segments
        if sector[0] != '':
            smoothness.append(int(sector[0]))
            complexity.append(int(sector[1]))
            orientation.append(int(sector[2]))

            smoothness.append(int(sector[3]))
            complexity.append(int(sector[4]))
            orientation.append(int(sector[5]))

            smoothness.append(int(sector[6]))
            complexity.append(int(sector[7]))
            orientation.append(int(sector[8]))

            smoothness.append(int(sector[9]))
            complexity.append(int(sector[10]))
            orientation.append(int(sector[11]))
       
        # road with 3 segments
        else:
            smoothness.append(int(sector[12]))
            complexity.append(int(sector[13]))
            orientation.append(int(sector[14]))

            smoothness.append(int(sector[15]))
            complexity.append(int(sector[16]))
            orientation.append(int(sector[17]))

            smoothness.append(int(sector[18]))
            complexity.append(int(sector[19]))
            orientation.append(int(sector[20]))


###################### DeepJanus ######################################

list_of_road = []
list_of_tag = []


# Input must be a CSV file that contains the id of the roads to plot. Roads are stored inside the db
with open(deepjanus_input_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # some refactoring added to rows for simplicity!      
        row = str(row).replace(";t", ",t").split(";")
        road = row[1].replace('"', '').replace('json.interactive_plot.json', 'json')
        list_of_road.append(road)            
        list_of_tag.append((road,row[3].replace('"', '').replace('\'', '')))


database_deepjanus = DBUtils(deepjanus_db)
roads_data_deepjanus = database_deepjanus.get_roads(list_of_road)

# sorting roads as they are in csv of tags
res = [next((sub for sub in roads_data_deepjanus  
       if sub[0] == i), 0) for i in list_of_road] 

# distinguish sectors of roads and create a new individual for each
for road_id, road_spine in res:
    roads = json.loads(road_spine)
    middle, left, right = get_geometry(roads)
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

    rv = RoadVisualizer(road_geometry, number_of_sectors=1)
    #rv._plot_sectors(road_id)
    #     plt.show()

    
    sector_num = 0
    for sector in rv.sectors:
        road_ids.append(road_id+ "_" +str(sector_num))
        sample_nodes = list()
        # define a new individual for each sector
        sector_geometry = [element for rs in sector for element in rs.geometry]
        middle_edge_x = [e['middle'][0] for e in sector_geometry]
        middle_edge_y = [e['middle'][1] for e in sector_geometry]
        middle_edge_z = [e['middle'][2] for e in sector_geometry]
        
        for point in zip(middle_edge_x, middle_edge_y, middle_edge_z):
            point_list = []
            point_list.append(point[0])
            point_list.append(point[1])
            point_list.append(point[2])
            sample_nodes.append(point_list)
        
        bbox_size = (-250.0, 0.0, 250.0, 500.0)
        road_bbox = RoadBoundingBox(bbox_size)
        res = BeamNGMember([], [tuple(t) for t in sample_nodes], len(sample_nodes), road_bbox)
        res.config = config
        res.problem = problem
        individual: BeamNGIndividual = BeamNGIndividual(res, config)
        individual.m.sample_nodes = us.new_resampling(individual.m.sample_nodes)
        deepjanus_roads.append(individual)
        sector_num += 1

# extract tags for each sector of each deepjanus road
for road in list_of_road:
    for _tag in list_of_tag:
        if _tag[0] == road:
            tag = _tag[1]
    # 1 sector
    pattern = re.compile('tag\-for\-sector\-0=\[smoothness\[([\d\.]+)\], complexity\[([\d\.]+)\], orientation\[([\d\.]+)\]]')
    
    sectors = pattern.findall(tag)  
    for sector in sectors:
        smoothness.append(int(sector[0]))
        complexity.append(int(sector[1]))
        orientation.append(int(sector[2]))

           
# measure metrics for each sector
for i in range(len(asfault_roads)): 
    x = asfault_roads[i]           

    #us.mean_lateral_position(x)

    min_radius = us.new_min_radius(x, 5)
    # curv_profile = us.entropy_of_curvature_profile(x)
    dir_coverage = us.direction_coverage(x)
    segment_count, segments = us.segment_count(x)
    # rel_seg = us.relative_segment_count(x)
    # segment_count_without_straight = us.segments_count_without_straight_segments(x)
    # rel_seg_without_straight = us.relative_segment_count_without_straight_segments(x)
    # segment_score = us.segments_score(x)

    minRadius.append(min_radius)
    directionCoverage.append(dir_coverage)
    # curvatureProfile.append(curv_profile)
    segmentCount.append(segment_count)
    # relSegment.append(rel_seg)
    # segmentCountWithoutStraight.append(segment_count_without_straight)
    # relSegmentWithoutStraight.append(rel_seg_without_straight)
    # segmentScore.append(segment_score)


for i in range(len(deepjanus_roads)): 
    x = deepjanus_roads[i]        
    # count, segments = us.segment_count(x)
    # print(count)
    # points_xx = []
    # points_yy = []
    # for point in x.m.sample_nodes:
    #     points_xx.append(point[0])
    #     points_yy.append(point[1])
    # [plt.plot(m, n, marker='.', color='red') for m,n in zip(points_xx, points_yy)]

    # i = 0
    # for segment in segments:
    #     points, angles = map(list,zip(*segment))
        
    #     points_x = []
    #     points_y = []
    #     for point in points:
    #         points_x.append(point[0])
    #         points_y.append(point[1])
    #     if i % 2 == 0:
    #         [plt.plot(m, n, marker='.', color='red') for m,n in zip(points_x, points_y)]
    #     else:
    #         [plt.plot(m, n, marker='.', color='blue') for m,n in zip(points_x, points_y)]

    #     i += 1
    

    #     if (i-1)%2 != 0:
    #         points_xxx = []
    #         points_yyy = []
    #         j = 0
    #         for i in range(len(points_xx)):
    #             if points_x[-1] == points_xx[i]:
    #                 j = i
    #                 break
    #         for jj in range(j, len(points_xx)):
    #             points_xxx.append(points_xx[jj])
    #             points_yyy.append(points_yy[jj])
    #         [plt.plot(m, n, marker='.', color='blue') for m,n in zip(points_xxx, points_yyy)]

    # plt.axes().set_aspect('equal', 'datalim')            
    # plt.show()
    #us.mean_lateral_position(x)

    min_radius = us.new_min_radius(x, 5)
    #curv_profile = us.entropy_of_curvature_profile(x)
    dir_coverage = us.direction_coverage(x)
    segment_count, segments = us.segment_count(x)
    # rel_seg = us.relative_segment_count(x)
    # segment_count_without_straight = us.segments_count_without_straight_segments(x)
    # rel_seg_without_straight = us.relative_segment_count_without_straight_segments(x)
    # segment_score = us.segments_score(x)

    minRadius.append(min_radius)
    directionCoverage.append(dir_coverage)
    #curvatureProfile.append(curv_profile)
    segmentCount.append(segment_count)
    # relSegment.append(rel_seg)
    # segmentCountWithoutStraight.append(segment_count_without_straight)
    # relSegmentWithoutStraight.append(rel_seg_without_straight)
    # segmentScore.append(segment_score)

# filename =  "correlation/correlation.csv"
# fw = open(filename, 'w')
# cf = csv.writer(fw, lineterminator='\n')

# #write the header
# cf.writerow(["road_id","smoothness", "complexity", "orientation", "min radius", "segment count", "dir coverage"])
# for i in range(0,len(road_ids)):
#     cf.writerow([road_ids[i], smoothness[i], complexity[i], orientation[i], minRadius[i], segmentCount[i], directionCoverage[i]])


smoothness_minRadius = stats.pearsonr(minRadius, smoothness)
#curvatureProfile_complexity = stats.pearsonr(curvatureProfile, complexity)
segmentCount_complexity = stats.pearsonr(segmentCount, complexity)
# relSegment_complexity = stats.pearsonr(relSegment, complexity)
# segmentCountWithoutStraight_complexity = stats.pearsonr(segmentCountWithoutStraight, complexity)
# relSegmentWithoutStraight_complexity = stats.pearsonr(relSegmentWithoutStraight, complexity)
# segmentScore_complexity = stats.pearsonr(segmentScore, complexity)
dirCoverage_orientation = stats.pearsonr(directionCoverage, orientation)

print("correlation & p-value:")
print(f"smoothness & minRadius: {smoothness_minRadius}")
#print(f"curvatureProfile & complexity: {curvatureProfile_complexity}")
print(f"segmentCount & complexity: {segmentCount_complexity}")
# print(f"relSegment & complexity: {relSegment_complexity}")
# print(f"segmentCountWithoutStraight & complexity: {segmentCountWithoutStraight_complexity}")
# print(f"relSegmentWithoutStraight & complexity: {relSegmentWithoutStraight_complexity}")
# print(f"segmentScore & complexity: {segmentScore_complexity}")
print(f"dirCoverage & orientation: {dirCoverage_orientation}")

