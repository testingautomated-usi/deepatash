from datetime import datetime
from pickle import POP
import sys
import random

from deap import base, creator, tools
import tensorflow as tf

# local
import config
from evaluator import Evaluator
from archive import Archive
from config import EXPECTED_LABEL, BITMAP_THRESHOLD, FEATURES, RUN_TIME, POPSIZE, GOAL,\
     RESEEDUPPERBOUND, MODEL, DIVERSITY_METRIC
from digit_mutator import DigitMutator
from sample import Sample
import utils as us
from utils import move_distance, bitmap_count, orientation_calc
import vectorization_tools


# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1.0))
# Define the individual.
creator.create("Individual", Sample, fitness=creator.FitnessMulti)

mnist = tf.keras.datasets.mnist
(X_train, y_train), (x_test, y_test) = mnist.load_data()
# Load the pre-trained model.
model = tf.keras.models.load_model(MODEL)
starting_seeds = []
evaluator = Evaluator()
dnn_pipeline = None
encoder = None

if DIVERSITY_METRIC == "LATENT":
    encoder = tf.keras.models.load_model("models/vae_encoder_test", compile=False)
    decoder = tf.keras.models.load_model("models/vae_decoder_test", compile=False)


def generate_initial_pop():
    samples = []
    for seed in range(len(x_test)):
        if y_test[seed] == EXPECTED_LABEL:
            starting_seeds.append(seed)
            xml_desc = vectorization_tools.vectorize(x_test[seed])
            sample = creator.Individual(xml_desc, EXPECTED_LABEL, seed)

            performance = evaluator.evaluate(sample, model)

            if performance > 0:
                misbehaviour = False
            else:
                misbehaviour = True

            predicted_label = sample.predicted_label

            sample_dict = {
                "expected_label": str(EXPECTED_LABEL),
                "features": {
                    "moves":  move_distance(sample),
                    "orientation": orientation_calc(sample,0),
                    "bitmaps": bitmap_count(sample, BITMAP_THRESHOLD)
                },
                "id": sample.id,
                "misbehaviour": misbehaviour,
                "performance": str(performance),
                "predicted_label": predicted_label,
                "seed": seed 
            }
            sample.from_dict(sample_dict) 
            samples.append(sample)
        
    return samples 

def reseed_individual(seeds):
    # Chooses randomly the seed among the ones that are not covered by the archive
    if len(starting_seeds) > len(seeds):
        seed = random.sample(set(starting_seeds) - seeds, 1)[0]
    else:
        seed = random.choice(starting_seeds)
    xml_desc = vectorization_tools.vectorize(x_test[seed])
    sample = creator.Individual(xml_desc, EXPECTED_LABEL, seed)
    return sample

def evaluate_individual(individual, features, goal, archive):
    evaluator.evaluate(individual, model)
    if DIVERSITY_METRIC == "LATENT":
        individual.compute_latent_vector(encoder)

    elif DIVERSITY_METRIC == "HEATMAP":
        individual.compute_explanation(model)

    evaluator.evaluate_sparseness(individual, archive.archive)

    # rescaled coordinates for distance calculation
    a = tuple()
    # original coordinates
    b = tuple()

    for ft in features:
        i = us.feature_simulator(ft[2], individual)
        a = a + (int(i/ft[1]),)
        b = b + (i,)

    individual.features = {
                        "moves":  move_distance(individual),
                        "orientation": orientation_calc(individual,0),
                        "bitmaps": bitmap_count(individual, BITMAP_THRESHOLD)
                    }
    
    individual.coordinate = b
    individual.distance_to_target = us.manhattan(a, goal)
    
    return individual.ff, individual.distance_to_target, individual.sparseness

def mutate_individual(individual):
    sample = DigitMutator(individual).mutate(x_test)
    return sample

def generate_features():
    features = []

    if "Moves" in FEATURES:
        f3 = ["moves", 5, "move_distance"]
        features.append(f3)
    if "Orientation" in FEATURES:
        f2 = ["orientation", 10, "orientation_calc"]
        features.append(f2)
    if "Bitmap" in FEATURES:
        f1 = ["bitmaps", 10, "bitmap_count"]
        features.append(f1)

    return features
       
def goal_acheived(population):
    for sample in population:
        if sample.distance_to_target == 0:
            print("Goal achieved")
            return True
    return False

toolbox.register("population", generate_initial_pop)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mutate", mutate_individual)
toolbox.register("select", tools.selRandom)

def main():
    start_time = datetime.now()
    # initialize archive
    archive = Archive()

    # Generate initial population and feature map
    print("initial population generation")   

    features = generate_features()

    population = toolbox.population()

    # Begin the generational process
    condition = True
    gen = 1

    # rescale GOAL to the feature intervals
    goal = list(GOAL)
    findex = 0
    for f in features:
        goal[findex] = int(GOAL[findex]/f[1])
        findex += 1
    goal = tuple(goal)

    while condition:
        print("Gen: ", gen)

        # Vary the population.
        offspring = [toolbox.clone(ind) for ind in population]

        # Reseeding
        if len(archive.archive) > 0:
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            candidate_seeds = archive.archived_seeds

            for i in range(seed_range):
                population[len(population) - i - 1] = reseed_individual(candidate_seeds)

            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    population[i] = reseed_individual(candidate_seeds)

        # Mutation.
        for ind in offspring:
            toolbox.mutate(ind)

        # Evaluate the individuals
        invalid_ind = [ind for ind in population + offspring]

        fitnesses = [toolbox.evaluate(i, features, goal, archive) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        # # random
        # pop = population + offspring
        # indexes = random.sample(range(0, len(pop)), POPSIZE)

        # population = []
        # for i in indexes:
        #     population.append(pop[i])
            
        # Update archive
        for ind in population:
            archive.update_archive(ind)

        gen += 1

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        if elapsed_time.seconds > RUN_TIME: #gen == GEN: #or goal_acheived(population):
            condition = False


    archive.export_archive("logs/"+str(name))
    print(len(archive.archive))
    return population


if __name__ == "__main__": 
    start_time = datetime.now()
    run = sys.argv[1]
    name = f"random_{run}-features_{FEATURES[0]}-{FEATURES[1]}-diversity_{DIVERSITY_METRIC}"
    pop = main()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("elapsed_time:", elapsed_time)
    print("GAME OVER")

    config.to_json("logs/"+str(name))
