from os.path import join
import json

# Dataset
EXPECTED_LABEL   = 5
# K-nearest
K = 1

GEN              = 1000
POPSIZE          = 100
IMG_SIZE         = 28
NUM_CLASSES      = 10
MODEL            = 'models/cnnClassifierTest.h5'
BITMAP_THRESHOLD = 0.5
FEATURES         = ["Orientation", "Moves"]#, "Orientation", "Moves", "Bitmaps"

move_range = 1
bitmaps_range = 4
orientation_range = 11


NUM_CELLS        = 25

XAI_METHOD       = "IG" # "IG" or "CEM"
GOAL             = (18, 130)



# goal cell for white area mov-lum (13, 174) or-lum (-30, 22) move-or (18, 130)  
# goal cell for gray area mov-lum (10, 124) or-lum (-30, 80) move-or (12, 90) 
# goal cell for dark area mov-lum (8, 160) or-lum (160, 60) move-or (0, 160)

DIVERSITY_METRIC = "LATENT" # "INPUT" "HEATMAP" "LATENT"
ARCHIVE_THRESHOLD = 0.01 # 4.8 for INPUT, 0.01 for LATENT, 0.09 IG HEATMAP
TARGET_THRESHOLD = 1


RESEEDUPPERBOUND = 10
MAX_BUCKET_SIZE = 50
RUN_TIME = 3600

# mutation operator probability
MUTOPPROB        = 0.5
MUTOFPROB        = 0.5
MUTUPPERBOUND    = 0.6
MUTLOWERBOUND    = 0.01




def to_json(folder):
    config = {
        'label': str(EXPECTED_LABEL),
        'image size': IMG_SIZE,
        'num classes' : NUM_CLASSES,
        'model': str(MODEL),
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive threshold': str(ARCHIVE_THRESHOLD), 
        'target cell': str(GOAL), 
        'move_range': move_range,
        'bitmaps_range': bitmaps_range,
        'orientation_range': orientation_range        
    }

    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
