from os.path import join
import json
from pathlib import Path
import os

BNG_USER = f"{str(Path.home())}/Documents/BeamNG.research"
BNG_HOME = f"{str(Path.home())}/Desktop/beamng/trunk" #os.environ['BNG_HOME']
GEN_RANDOM = 'GEN_RANDOM'
GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'
GEN_DIVERSITY = 'GEN_DIVERSITY'

SEG_LENGTH = 25
NUM_SPLINE_NODES =10
INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
ROAD_BBOX_SIZE = (-1000, 0, 1000, 1500)
EXECTIME = 0
INVALID = 0


# K-nearest
K = 1

GEN              = 1000
POPSIZE          = 10


FEATURES         = ["MeanLateralPosition", "SegmentCount"] # Curvature, SDSteeringAngle, SegmentCount, MeanLateralPosition
NUM_CELLS        = 25


mlp_range       = 2
curv_range      = 1
sdstd_range     = 7
turncnt_range   = 1

GOAL             = (150, 4)
# goal cell for white area MLP-TurnCnt (150, 4)
# goal cell for gray area  MLP-StdSA (168, 90), mlp-Curv (165, 14), mlp-TurnCnt (166, 2)
# goal cell for dark area  MLP-TurnCnt (160, 3), MLP-StdSA (162,108), Curv-StdSA (22, 75), MLP-Curv (167, 20)

DIVERSITY_METRIC = "INPUT" 
ARCHIVE_THRESHOLD =  10.0 #35.0 

RESEEDUPPERBOUND = 2
MAX_BUCKET_SIZE = 50
RUN_TIME = 36000
TARGET_THRESHOLD = 2

def to_json(folder):
    config = {
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive threshold': str(ARCHIVE_THRESHOLD), 
        'target cell': str(GOAL)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
