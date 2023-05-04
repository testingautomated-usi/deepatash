from os.path import join
import json

# Dataset
EXPECTED_LABEL   = 1
# K-nearest
K = 1

INITIAL_SEED       = 1000
POPSIZE          = 100

VOCAB_SIZE      = 10000
INPUT_MAXLEN        = 2000
NUM_CLASSES      = 10
MODEL            = 'models/text_classifier.h5'
BITMAP_THRESHOLD = 0.5
FEATURES         = ["NegCount", "VerbCount"]#, "NegCount", "PosCount", "VerbCount"


NUM_CELLS        = 25

GOAL             = (4, 11)

DISTANCE        = 5

# goal cell for white area  neg-pos (11, 6) pos-verb (11, 13) neg-verb (11, 6)
# goal cell for grey area  neg-pos (14, 6) pos-verb (8, 12) neg-verb (9, 6)
# goal cell for dark area neg-pos (11, 8) pos-verb (1, 7) neg-verb (4, 12)

DIVERSITY_METRIC = "HEATMAP" # "INPUT" "HEATMAP" "LATENT"
TARGET_THRESHOLD = 1

TARGET_SIZE     = 42

RESEEDUPPERBOUND = 10
RUN_TIME = 3600



META_FILE       = "../experiments/data/imdb/DeepHyperion/meta.json"


def to_json(folder):
    config = {
        'initial seed': str(INITIAL_SEED),
        'label': str(EXPECTED_LABEL),
        'Vocab size': VOCAB_SIZE,
        'Input MaxLen': INPUT_MAXLEN,
        'num classes' : NUM_CLASSES,
        'model': str(MODEL),
        'features': str(FEATURES),
        'pop size': str(POPSIZE),
        'diversity': str(DIVERSITY_METRIC),
        'archive size': str(TARGET_SIZE), 
        'target cell': str(GOAL),        
    }

    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
