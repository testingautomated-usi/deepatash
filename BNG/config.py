from os.path import join
import json
from pathlib import Path


model_file = 'data/trained_models_colab/self-driving-car-178-2020.h5'

BNG_USER = f"{str(Path.home())}/Documents/BeamNG.research"
BNG_HOME = f"{str(Path.home())}/Desktop/beamng/trunk" #os.environ['BNG_HOME']
GEN_RANDOM = 'GEN_RANDOM'
GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'
GEN_DIVERSITY = 'GEN_DIVERSITY'

SEG_LENGTH = 25
NUM_SPLINE_NODES = 10
INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
ROAD_BBOX_SIZE = (-1000, 0, 1000, 1500)
EXECTIME = 0
INVALID = 0

INITIAL_SEED    = 80
INITIAL_POP     = 48

TARGET_SIZE     = 10

# K-nearest
K = 1

GEN              = 1000
POPSIZE          = 10

FEATURES         = ["Curvature", "MeanLateralPosition"] # Curvature, SDSteeringAngle, SegmentCount, MeanLateralPosition
NUM_CELLS        = 25

APPROACH        = "ga" # ["ga", "nsga2"]

RUN_ID          = 5

GOAL             = (23, 9)

# goal cell for white area Curv-MLP (23, 9), Curv-TurnCnt (24, 6), TurnCnt-StdSA (8, 10)
# goal cell for grey area Curv-MLP (21, 8), Curv-TurnCnt (20, 8), TurnCnt-StdSA (7, 17)
# goal cell for dark area Curv-MLP (22, 7), Curv-TurnCnt (23, 4), -

DIVERSITY_METRIC = "INPUT" 

META_FILE       = "../experiments/data/bng/DeepHyperion/meta.json"

NUM_EXPERIMENTS = 10

RESEEDUPPERBOUND = 2

RUN_TIME = 36000
TARGET_THRESHOLD = 1

NUM_RETRAIN = 10

def to_json(folder):
    config = {
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive size': str(TARGET_SIZE), 
        'target cell': str(GOAL),
        'run id': str(RUN_ID),
        'run time': str(RUN_TIME),
        'approach': str(APPROACH)
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
