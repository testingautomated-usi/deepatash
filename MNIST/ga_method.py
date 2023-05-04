import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
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
from config import EXPECTED_LABEL, BITMAP_THRESHOLD, FEATURES, POPSIZE, GOAL, RESEEDUPPERBOUND, MODEL, DIVERSITY_METRIC, TARGET_SIZE, META_FILE 
from digit_mutator import DigitMutator
from sample import Sample
import utils as us
from utils import move_distance, bitmap_count, orientation_calc
import vectorization_tools
from feature import Feature
from timer import Timer

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

    # diversity computation
    individual.sparesness, _ = evaluator.evaluate_sparseness(individual, archive.archive)

    if individual.distance_to_target == None:         
        # original coordinates
        b = tuple()

        individual.features = {
                    "moves":  move_distance(individual),
                    "orientation": orientation_calc(individual,0),
                    "bitmaps": bitmap_count(individual, BITMAP_THRESHOLD)
        }

        for ft in features:
            i = ft.get_coordinate_for(individual)
            if i != None:
                b = b + (i,)
            else:
                # this individual is out of range and must be discarded
                individual.distance_to_target = np.inf
                return (np.inf, )
        
        individual.coordinate = b
        individual.distance_to_target = us.manhattan(b, goal)
    
    return (individual.distance_to_target,)


def mutate_individual(individual):
    sample = DigitMutator(individual).mutate(x_test)
    return sample


def generate_features(meta_file):
    features = []
    with open(meta_file, 'r') as f:
        meta = json.load(f)["features"]
        
    if "Moves" in FEATURES:
        f3 = Feature("moves", meta["moves"]["min"], meta["moves"]["max"], "move_distance", 25)
        features.append(f3)
    if "Orientation" in FEATURES:
        f2 = Feature("orientation",meta["orientation"]["min"], meta["orientation"]["max"], "orientation_calc", 25)
        features.append(f2)
    if "Bitmaps" in FEATURES:
        f1 = Feature("bitmaps",meta["bitmaps"]["min"], meta["bitmaps"]["max"], "bitmap_count", 25)
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
toolbox.register("select", tools.selBest)  
toolbox.register("mutate", mutate_individual)

def evaluate_batch(individuals):
    ind_batch = []
    latent_batch = []
    heatmap_batch = []

    for individual in individuals:
        if individual.predicted_label == None:
            ind_batch.append(individual)
        if individual.latent_vector is None:
            latent_batch.append(individual)
        if individual.explanation is None:
            heatmap_batch.append(individual)

    if len(ind_batch) > 0:
        evaluator.evaluate_batch(ind_batch, model)
    if DIVERSITY_METRIC == "LATENT" and len(latent_batch) > 0:
        evaluator.evaluate_latent_batch(latent_batch, encoder)
    if DIVERSITY_METRIC == "HEATMAP" and len(heatmap_batch) > 0:
        evaluator.evaluate_heatmap_batch(heatmap_batch)

def main():
    # initialize archive
    archive = Archive(TARGET_SIZE)

    features = generate_features(META_FILE)


    # Generate initial population and feature map 
    log.info(f"Generate initial population")  
           

    population = toolbox.population()

    # Evaluate the individuals
    invalid_ind = [ind for ind in population]

    evaluate_batch(invalid_ind)

    fitnesses = [toolbox.evaluate(i, features, GOAL, archive) for i in invalid_ind]

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the next generation population
    population = toolbox.select(population, POPSIZE)


    # Begin the generational process
    gen = 1

    while Timer.has_budget():
        log.info(f"Iteration: {gen}")

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
            log.info(f"Archive seeds: {candidate_seeds}")

            for i in range(seed_range):
                ind =  population[len(population) - i - 1]
                new_ind = reseed_individual(candidate_seeds)
                population[len(population) - i - 1] = new_ind
                log.info(f"ind {ind.id} with seed {ind.seed} reseeded by {new_ind.id} with seed {new_ind.seed}")

            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    ind =  population[i]
                    new_ind = reseed_individual(candidate_seeds)
                    population[i] = new_ind
                    log.info(f"ind {ind.id} with seed {ind.seed} reseeded by {new_ind.id} with seed {new_ind.seed}")

        # Evaluate the individuals
        invalid_ind = [ind for ind in offspring + population]

        evaluate_batch(invalid_ind)

        fitnesses = [toolbox.evaluate(i, features, GOAL, archive) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Update archive
        for ind in offspring + population:
            is_added = archive.update_archive(ind, evaluator)
            # if a new ind added to archive, recompute all the sparsenesses
            if is_added:
                archive.recompute_sparseness_archive(evaluator)

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        for individual in population:
            log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['moves']}, {individual.features['orientation']}, {individual.features['bitmaps']}), performance {individual.ff} and distance {individual.distance_to_target} selected")

        gen += 1
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

    

