from os.path import join
import json

# Dataset
EXPECTED_LABEL   = 5
# K-nearest
K = 1

POPSIZE          = 100

IMG_SIZE         = 28
NUM_CLASSES      = 10
MODEL            = 'models/mnist_classifier.h5'
BITMAP_THRESHOLD = 0.5
FEATURES         = ["Orientation", "Bitmaps"]   # ["Orientation", "Moves", "Bitmaps"]


NUM_CELLS        = 25

GOAL             = (4, 1)
RUN_ID           = 1

# these goal cells computed from 10 times of running DeepHyperion:
# goal cell for white area mov-lum (11, 3)  or-lum (10, 2) move-or  (17, 10)
# goal cell for grey area mov-lum (21, 9) or-lum (19, 4) move-or (16, 11)
# goal cell for dark area mov-lum (6, 0) or-lum (4, 1) move-or (7, 5)

DIVERSITY_METRIC =  "LATENT" # ["INPUT", "HEATMAP", "LATENT"] 

APPROACH        = "nsga2" # ["ga", "nsga2"]

TARGET_THRESHOLD = 1 # closeness to the target 

TARGET_SIZE     = 81 # target archive size

RESEEDUPPERBOUND = 10

RUN_TIME = 600

# mutation operator probability
MUTOPPROB        = 0.5
MUTOFPROB        = 0.5
MUTUPPERBOUND    = 0.6
MUTLOWERBOUND    = 0.01

META_FILE       = "../experiments/data/mnist/DeepHyperion/meta.json"


NUM_RETRAIN     = 1 # 10



def to_json(folder):
    config = {
        'label': str(EXPECTED_LABEL),
        'image size': IMG_SIZE,
        'num classes' : NUM_CLASSES,
        'model': str(MODEL),
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
