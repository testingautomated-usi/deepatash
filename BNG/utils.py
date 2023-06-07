import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging as log
import sys

import tensorflow as tf

#from tensorflow import keras
from config import DIVERSITY_METRIC
import numpy as np
# local imports
from skimage.color import rgb2gray
from core.curvature import define_circle
from self_driving.edit_distance_polyline import _calc_dist_angle
from itertools import tee
from shapely.geometry import LineString, Polygon, Point

THE_NORTH = [0,1]
ANGLE_THRESHOLD = 0.005

import math
import numpy as np

NAMESPACE = '{http://www.w3.org/2000/svg}'

def time_to_sec(t):
   h, m, s = map(float, str(t).split(':'))
   return h * 3600 + m * 60 + s


def to_gray_uint(image):
    return np.uint8(rgb2gray(image) * 255)

def get_element_by_seed(fm, seed):
    for (x,y), value in np.ndenumerate(fm):
        if value != None:
            for v in value:
                if v.seed == seed:
                    return (x,y)
    return None

def get_distance(ind1, ind2):
    """ Computes distance based on configuration """

    if DIVERSITY_METRIC == "INPUT":
        # input space
        distance = ind1.ind.m.distance(ind2.ind.m)
       

    return distance



def get_distance_by_metric(ind1, ind2, metric):
    """ Computes distance based on metric """

    if metric == "INPUT":
        # input space
        distance = ind1.ind.m.distance(ind2.ind.m)
    

    return distance

def euclidean(img1, img2):
    dist = np.linalg.norm(img1 - img2)
    return dist

def manhattan(coords_ind1, coords_ind2):
    return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])

def feature_simulator(function, x):
        """
        Calculates the number of control points of x's svg path/number of bitmaps above threshold
        :param x: genotype of candidate solution x
        :return:
        """
        if function == 'min_radius':
            return new_min_radius(x)
        if function == 'mean_lateral_position':
            return mean_lateral_position(x)
        if function == "dir_coverage":
            return direction_coverage(x)
        if function == "segment_count":
            return segment_count(x)
        if function == "sd_steering":
            return sd_steering(x)
        if function == "curvature":
            return curvature(x)


def segment_count(x):
    nodes = x.ind.m.sample_nodes
    
    count , segments = identify_segment(nodes)
    return count #, segments
    # TODO Note that this is identify_segments with a final 's'
    # segments = identify_segments(nodes)
    # return len(segments), segments

def rel_segment_count(x):
    nodes = x.ind.m.sample_nodes
    
    count, segments = identify_segment(nodes)
    rel = (count/len(nodes))
    rel = rel/0.04093567251461988
    return int(rel*100) #, segments

# counts only turns, split turns
def identify_segment(nodes):
     # result is angle, distance, [x2,y2], [x1,y1]
     result, _ = _calc_dist_angle(nodes)

     segments = []
     SEGMENT_THRESHOLD = 15
     SEGMENT_THRESHOLD2 = 10
     ANGLE_THRESHOLD = 0.005


     # iterate over the nodes to get the turns bigger than the threshold
     # a turn category is assigned to each node
     # l is a left turn
     # r is a right turn
     # s is a straight segment
     # TODO: first node is always a s
     turns = []
     for i in range(0, len(result)):
         # result[i][0] is the angle
         angle_1 = (result[i][0] + 180) % 360 - 180
         if np.abs(angle_1) > ANGLE_THRESHOLD:
             if(angle_1) > 0:
                 turns.append("l")
             else:
                 turns.append("r")
         else:
             turns.append("s")

     # this generator groups the points belonging to the same category
     def grouper(iterable):
         prev = None
         group = []
         for item in iterable:
             if not prev or item == prev:
                 group.append(item)
             else:
                 yield group
                 group = [item]
             prev = item
         if group:
             yield group

     # this generator groups:
     # - groups of points belonging to the same category
     # - groups smaller than 10 elements
     def supergrouper1(iterable):
         prev = None
         group = []
         for item in iterable:
             if not prev:
                 group.extend(item)
             elif len(item) < SEGMENT_THRESHOLD2 and item[0] == "s":
                 item = [prev[-1]] * len(item)
                 group.extend(item)
             elif len(item) < SEGMENT_THRESHOLD and item[0] != "s" and prev[-1] == item[0]:
                 item = [prev[-1]] * len(item)
                 group.extend(item)
             else:
                 yield group
                 group = item
             prev = item
         if group:
             yield group

     # this generator groups:
     # - groups of points belonging to the same category
     # - groups smaller than 10 elements
     def supergrouper2(iterable):
         prev = None
         group = []
         for item in iterable:
             if not prev:
                 group.extend(item)
             elif len(item) < SEGMENT_THRESHOLD:
                 item = [prev[-1]]*len(item)
                 group.extend(item)
             else:
                 yield group
                 group = item
             prev = item
         if group:
             yield group

     groups = grouper(turns)

     supergroups1 = supergrouper1(groups)

     supergroups2 = supergrouper2(supergroups1)

     count = 0
     segment_indexes = []
     segment_count = 0
     for g in supergroups2:
        if g[-1] != "s":
            segment_count += 1
        # TODO
        #count += (len(g) - 1)
        count += (len(g))
        # TODO: count -1?
        segment_indexes.append(count)

     # TODO
     #segment_indexes.append(len(turns) - 1)

     segment_begin = 0
     for idx in segment_indexes:
         segment = []
         #segment_end = idx + 1
         segment_end = idx
         for j in range(segment_begin, segment_end):
             if j == 0:
                 segment.append([result[j][2][0], result[j][0]])
             segment.append([result[j][2][1], result[j][0]])
         segment_begin = segment_end
         segments.append(segment)

     return segment_count, segments


# https://docs.python.org/3/library/itertools.html
# Itertools Recipes
def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def direction_coverage(x, n_bins=25):
    """Measure the coverage of road directions w.r.t. to the North (0,1) using the control points of the given road
    to approximate the road direction. BY default we use 36 bins to have bins of 10 deg each"""
    # Note that we need n_bins+1 because the interval for each bean is defined by 2 points
    coverage_buckets = np.linspace(0.0, 360.0, num=n_bins+1)
    direction_list = []
    for a, b in _pairwise(x.ind.m.sample_nodes):
        # Compute the direction of the segment defined by the two points
        road_direction = [b[0] - a[0], b[1] - a[1]]
        # Compute the angle between THE_NORTH and the road_direction.
        # E.g. see: https://www.quora.com/What-is-the-angle-between-the-vector-A-2i+3j-and-y-axis
        # https://www.kite.com/python/answers/how-to-get-the-angle-between-two-vectors-in-python
        unit_vector_1 = road_direction/np.linalg.norm(road_direction)
        dot_product = np.dot(unit_vector_1, THE_NORTH)
        angle = math.degrees(np.arccos(dot_product))
        direction_list.append(angle)

    # Place observations in bins and get the covered bins without repetition
    covered_elements = set(np.digitize(direction_list, coverage_buckets))
    return int((len(covered_elements) / len(coverage_buckets))*100)


def new_min_radius(x, w=5):
    mr = np.inf
    mincurv = []
    nodes = x.ind.m.sample_nodes
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w-1)/2)]
        p3 = nodes[i + (w-1)]
        #radius = findCircle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
        radius = define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
            mincurv = [p1, p2, p3]

    if mr  > 90:
        mr = 90

    return int(mr*3.280839895)#, mincurv


def curvature(x, w=5):
    mr = np.inf
    mincurv = []
    nodes = x.ind.m.sample_nodes
    for i in range(len(nodes) - w):
        p1 = nodes[i]
        p2 = nodes[i + int((w-1)/2)]
        p3 = nodes[i + (w-1)]
        #radius = findCircle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
        radius = define_circle(p1, p2, p3)
        if radius < mr:
            mr = radius
            mincurv = [p1, p2, p3]

    curvature = (1/mr)*100

    return int(curvature)#, mincurv

def sd_steering(x):
    states = x.ind.m.simulation.states
    steering = []
    for state in states:
        steering.append(state.steering)
    sd_steering = np.std(steering)
    return int(sd_steering)


def mean_lateral_position(x):
    states = x.ind.m.simulation.states
    lp = []
    for state in states:
        lp.append(state.oob_distance)
    mean_lp = np.mean(lp) * 100
    return int(mean_lp)


def log_exception(extype, value, trace):
    log.exception('Uncaught exception:', exc_info=(extype, value, trace))

def setup_logging(log_file):
    file_handler = log.FileHandler(log_file, 'a', 'utf-8')
    term_handler = log.StreamHandler()
    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                  level=log.INFO, handlers=[term_handler, file_handler])
    sys.excepthook = log_exception
    log.info('Started the logging framework writing to file: %s', log_file)



def calc_point_edges(p1, p2):
    origin = np.array(p1[0:2])

    a = np.subtract(p2[0:2], origin)
    # print(p1, p2)
    v = (a / np.linalg.norm(a)) * p1[3] / 2

    l = origin + np.array([-v[1], v[0]])
    r = origin + np.array([v[1], -v[0]])
    return tuple(l), tuple(r)


def get_geometry(middle_nodes):
    middle = []
    right = []
    left = []
    n = len(middle) + len(middle_nodes)

    # add middle nodes (road points): adds central spline
    middle += list(middle_nodes)
    left += [None] * len(middle_nodes)
    right += [None] * len(middle_nodes)

    # recalculate nodes: adds points of the lateral lane margins
    for i in range(n - 1):
        l, r = calc_point_edges(middle[i], middle[i + 1])
        left[i] = l
        right[i] = r
    # the last middle point
    right[-1], left[-1] = calc_point_edges(middle[-1], middle[-2])

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

        road_geometry.append(point)

    return road_geometry


def is_oob(road_nodes, simulation_states):
    # Create the road geometry from the nodes. At this point nodes have been reversed already if needed.
    road_geometry = get_geometry(road_nodes)

    # Compute right polygon
    # Create the right lane polygon from the geometry
    left_edge_x = np.array([e['middle'][0] for e in road_geometry])
    left_edge_y = np.array([e['middle'][1] for e in road_geometry])
    right_edge_x = np.array([e['right'][0] for e in road_geometry])
    right_edge_y = np.array([e['right'][1] for e in road_geometry])

    # Compute the "short" edge at the end of the road to rule out false positives
    shorty = LineString([(left_edge_x[-1], left_edge_y[-1]), (right_edge_x[-1], right_edge_y[-1])]).buffer(2.0)

    # Note that one must be in reverse order for the polygon to close correctly
    right_edge = LineString(zip(right_edge_x[::-1], right_edge_y[::-1]))
    left_edge = LineString(zip(left_edge_x, left_edge_y))

    l_edge = left_edge.coords
    r_edge = right_edge.coords

    right_lane_polygon = Polygon(list(l_edge) + list(r_edge))

    #middle = [e['middle'] for e in road_geometry]
    #right = [e['right'] for e in road_geometry]
    #road_poly = [(p[0], p[1]) for p in middle]
    #right = [(p[0], p[1]) for p in right]
    #road_poly.extend(right[::-1])
    #right_polygon = Polygon(road_poly)


    first_oob_state = None
    position_of_first_oob_state = None
    for idx, simulation_state in enumerate(simulation_states):
        position = Point(simulation_state["pos"][0], simulation_state["pos"][1])
        if not right_lane_polygon.contains(position):

            # As soon as an observation is outside the lane polygon we mark the OOB, and return that position. All the
            # subsequent states will be removed/discarded
            log.debug("First OOB state found at %d", idx)
            first_oob_state = idx
            position_of_first_oob_state = position

            break

    if first_oob_state is not None:
        if shorty.contains(position_of_first_oob_state):
            log.info("*    False Positive. Short Edge")
            return False, None
        else:
            return True, first_oob_state
    else:
        return False, None