import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging as log
import sys

import tensorflow as tf

#from tensorflow import keras
from config import BITMAP_THRESHOLD, DIVERSITY_METRIC
import numpy as np
# local imports
from skimage.color import rgb2gray
from itertools import tee

IMG_SIZE = 28

import math
from sklearn.linear_model import LinearRegression
import re
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np

NAMESPACE = '{http://www.w3.org/2000/svg}'


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
        distance = euclidean(ind1.purified, ind2.purified)
    
    elif DIVERSITY_METRIC == "LATENT":
        # latent space
        distance = euclidean(ind1.latent_vector, ind2.latent_vector)


    elif DIVERSITY_METRIC == "HEATMAP":
        # heatmap space
        distance = euclidean(ind1.explanation, ind2.explanation)



    return distance



def get_distance_by_metric(ind1, ind2, metric):
    """ Computes distance based on metric """

    if metric == "INPUT":
        # input space
        distance = euclidean(ind1.purified, ind2.purified)
    
    elif metric == "LATENT":
        # latent space
        distance = euclidean(ind1.latent_vector, ind2.latent_vector)


    elif metric == "HEATMAP":
        # heatmap space
        distance = euclidean(ind1.explanation, ind2.explanation)



    return distance

def kl_divergence(ind1, ind2):
    mu1 = ind1[0]
    sigma_1 = ind1[1]
    mu2 = ind2[0]
    sigma_2 = ind2[1]

    sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
    sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2
    sigma_diag_2_inv = np.linalg.inv(sigma_diag_2)

    kl = 0.5 * (np.log(np.linalg.det(sigma_diag_2) / np.linalg.det(sigma_diag_2)) 
        - mu1.shape[0] + np.trace(np.matmul(sigma_diag_2_inv, sigma_diag_1))  
        + np.matmul(np.matmul(np.transpose(mu2 - mu1),sigma_diag_2_inv), (mu2 - mu1)))
    return kl

def euclidean(img1, img2):
    dist = np.linalg.norm(img1 - img2)
    return dist

def manhattan(coords_ind1, coords_ind2):
    return abs(coords_ind1[0] - coords_ind2[0]) + abs(coords_ind1[1] - coords_ind2[1])

def feature_simulator(function, x):
    """
    Calculates the value of the desired feature
    :param function: name of the method to compute the feature value
    :param x: genotype of candidate solution x
    :return: feature value
    """
    if function == 'bitmap_count':
        return bitmap_count(x, BITMAP_THRESHOLD)
    if function == 'move_distance':
        return move_distance(x)
    if function == 'orientation_calc':
        return orientation_calc(x, 0)

def bitmap_count(digit, threshold):
    image = deepcopy(digit.purified)
    bw = np.asarray(image)
    #bw = bw / 255.0
    count = 0
    for x in np.nditer(bw):
        if x > threshold:
            count += 1
    return count

def move_distance(digit):
    root = ET.fromstring(digit.xml_desc)
    svg_path = root.find(NAMESPACE + 'path').get('d')
    pattern = re.compile('([\d\.]+),([\d\.]+)\sM\s([\d\.]+),([\d\.]+)')
    segments = pattern.findall(svg_path)
    if len(segments) > 0:
        dists = [] # distances of moves
        for segment in segments:
            x1 = float(segment[0])
            y1 = float(segment[1])
            x2 = float(segment[2])
            y2 = float(segment[3])
            dist = math.sqrt(((x1-x2)**2)+((y1-y2)**2))
            dists.append(dist)
        return int(np.sum(dists))
    else:
        return 0

def orientation_calc(digit, threshold):
    x = []
    y = []
    image = deepcopy(digit.purified)
    bw = np.asarray(image)
    for iz, ix, iy, ig in np.ndindex(bw.shape):
        if bw[iz, ix, iy, ig] > threshold:
            x.append([iy])
            y.append(ix)
    if len(x)!= 0:
        X = np.array(x)
        Y = np.array(y)
        lr = LinearRegression(fit_intercept=True).fit(X, Y)
        normalized_ori = -lr.coef_ 
        new_ori = normalized_ori * 100
        return int(new_ori)
    else:
        return 0



# Useful function that shapes the input in the format accepted by the ML model.
def reshape(v):
    v = (np.expand_dims(v, 0))
    # Shape numpy vectors
    if tf.keras.backend.image_data_format() == 'channels_first':
        print(here)
        v = v.reshape(v.shape[0], 1, IMG_SIZE, IMG_SIZE)
    else:
        v = v.reshape(v.shape[0], IMG_SIZE, IMG_SIZE, 1)
    v = v.astype('float32')
    v = v / 255.0
    return v

def setup_logging(log_to, debug):

    def log_exception(extype, value, trace):
        log.exception('Uncaught exception:', exc_info=(extype, value, trace))

    # Disable annoyng messages from matplot lib.
    # See: https://stackoverflow.com/questions/56618739/matplotlib-throws-warning-message-because-of-findfont-python
    log.getLogger('matplotlib.font_manager').disabled = True

    term_handler = log.StreamHandler()
    log_handlers = [term_handler]
    start_msg = "Started test generation"

    if log_to is not None:
        file_handler = log.FileHandler(log_to, 'a', 'utf-8')
        log_handlers.append( file_handler )
        start_msg += " ".join(["writing to file: ", str(log_to)])

    log_level = log.DEBUG if debug else log.INFO

    log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=log_level, handlers=log_handlers)

    sys.excepthook = log_exception

    log.info(start_msg)