import random
from datetime import datetime
from pickle import POP
import sys
import random as rand
from deap import base, creator, tools
from evaluator import Evaluator
import tensorflow as tf
import logging as log
from pathlib import Path
import numpy as np

# local
import config
from archive import Archive
from config import EXPECTED_LABEL, BITMAP_THRESHOLD, FEATURES, RUN_TIME, POPSIZE, GOAL, RESEEDUPPERBOUND, MODEL, DIVERSITY_METRIC, XAI_METHOD
from digit_mutator import DigitMutator
from sample import Sample
import utils as us
from utils import move_distance, bitmap_count, orientation_calc
import vectorization_tools


# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Define the individual.
creator.create("Individual", Sample, fitness=creator.FitnessMin)

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
    if individual.predicted_label == None:
        evaluator.evaluate(individual, model)
    if DIVERSITY_METRIC == "LATENT" and individual.latent_vector is None:
        individual.compute_latent_vector(encoder)

    elif DIVERSITY_METRIC == "HEATMAP" and individual.explanation is None:
        individual.compute_explanation()

    # diversity computation
    evaluator.evaluate_sparseness(individual, archive.archive)

    if individual.distance_to_target == np.inf:         
        # rescaled coordinates for distance calculation
        a = tuple()
        # original coordinates
        b = tuple()

        for ft in features:
            i = us.feature_simulator(ft[2], individual)
            a = a + ((i/ft[1]),)
            b = b + (i,)

        individual.features = {
                    "moves":  move_distance(individual),
                    "orientation": orientation_calc(individual,0),
                    "bitmaps": bitmap_count(individual, BITMAP_THRESHOLD)
        }
        
        individual.coordinate = b
        individual.distance_to_target = us.manhattan(a, goal)
    
    log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['moves']}, {individual.features['orientation']}, {individual.features['bitmaps']}) and distance {individual.distance_to_target} evaluated")


    return (individual.distance_to_target,)

def mutate_individual(individual):
    sample = DigitMutator(individual).mutate(x_test)
    return sample

def generate_features():
    features = []

    if "Moves" in FEATURES:
        f3 = ["moves", 1, "move_distance"]
        features.append(f3)
    if "Orientation" in FEATURES:
        f2 = ["orientation", 11, "orientation_calc"]
        features.append(f2)
    if "Bitmaps" in FEATURES:
        f1 = ["bitmaps", 4, "bitmap_count"]
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
toolbox.register("select", tools.selBest) # tools.selTournament, tournsize=2) 
toolbox.register("mutate", mutate_individual)



def main():
    start_time = datetime.now()
    # initialize archive
    archive = Archive()

    features = generate_features()

    # rescale GOAL to the feature intervals
    goal = list(GOAL)
    findex = 0
    for f in features:
        goal[findex] = (GOAL[findex]/f[1])
        findex += 1
    goal = tuple(goal)


    # Generate initial population and feature map 
    log.info(f"Generate initial population")  
           

    population = toolbox.population()

    # Evaluate the individuals
    invalid_ind = [ind for ind in population]

    fitnesses = [toolbox.evaluate(i, features, goal, archive) for i in invalid_ind]

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the next generation population
    population = toolbox.select(population, POPSIZE)


    # Begin the generational process
    condition = True
    gen = 1

    while condition:
        log.info(f"Iteration: {gen}")

        # Vary the population.
        # offspring = [toolbox.clone(ind) for ind in population]
        offspring = []
        for ind in population:
            sample = creator.Individual(ind.xml_desc, EXPECTED_LABEL, ind.seed)
            offspring.append(sample)
                
        # Mutation.
        log.info("Mutation")
        for ind in offspring:
            toolbox.mutate(ind)

        # Reseeding
        if len(archive.archive) > 0:
            log.info(f"Reseeding")
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            candidate_seeds = archive.archived_seeds

            for i in range(seed_range):
                population[len(population) - i - 1] = reseed_individual(candidate_seeds)

            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    population[i] = reseed_individual(candidate_seeds)

        # Evaluate the individuals
        invalid_ind = [ind for ind in offspring + population]

        fitnesses = [toolbox.evaluate(i, features, goal, archive) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Update archive
        for ind in offspring + population:
            archive.update_archive(ind, evaluator)

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        for individual in population:
            log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['moves']}, {individual.features['orientation']}, {individual.features['bitmaps']}) and distance {individual.distance_to_target} selected")


        gen += 1

        end_time = datetime.now()
        elapsed_time = end_time - start_time
        if elapsed_time.seconds > RUN_TIME: #gen == GEN: #or goal_acheived(population):
            condition = False
        log.info(f"Archive: {len(archive.archive)}")

    log.info(f"Archive: {len(archive.archive)}")
    archive.export_archive(name)
    return population


if __name__ == "__main__": 
    start_time = datetime.now()
    run = sys.argv[1]
    name = f"logs/{run}-ga_-features_{FEATURES[0]}-{FEATURES[1]}-diversity_{DIVERSITY_METRIC}"

    Path(name).mkdir(parents=True, exist_ok=True)
    config.to_json(name)
    
    log_to = f"{name}/logs.txt"
    debug = f"{name}/debug.txt"

    # Setup logging
    us.setup_logging(log_to, debug)
    print("Logging results to " + log_to)

    pop = main()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    log.info("elapsed_time:" + str(elapsed_time))
    print("GAME OVER")

    

