import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import random
from datetime import datetime
import sys
from deap import base, creator, tools
from evaluator import Evaluator
import logging as log
from pathlib import Path
import numpy as np
from datasets import load_dataset
from gensim.models.doc2vec import Doc2Vec

# local
import config
from archive import Archive
from config import EXPECTED_LABEL, FEATURES, POPSIZE, GOAL, RESEEDUPPERBOUND, DIVERSITY_METRIC, TARGET_SIZE, META_FILE 
from text_mutator import TextMutator
from sample import Sample
import utils as us
from utils import count_neg, count_pos, count_verbs
from feature import Feature
from timer import Timer
from config import INITIAL_SEED


# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Define the individual.
creator.create("Individual", Sample, fitness=creator.FitnessMin)

DATASET_DIR = "data"
test_ds = load_dataset('imdb', cache_dir=f"{DATASET_DIR}/imdb", split='test')
x_test, y_test = test_ds['text'], test_ds['label']


starting_seeds = []
evaluator = Evaluator()


if DIVERSITY_METRIC == "LATENT":
    doc2vec = Doc2Vec.load("models/Doc2Vec_dbow,d100,n5,mc2,t16")


def generate_initial_pop(features, goal):
    samples = []
    for seed in range(len(x_test)):
        if y_test[seed] == EXPECTED_LABEL:
            starting_seeds.append(seed)

    random.shuffle(starting_seeds)
    for seed in starting_seeds:
        b = tuple()
        individual = creator.Individual(x_test[seed], EXPECTED_LABEL, seed) 
        individual.features = {
                    "negcount":  count_neg(individual.text),
                    "poscount": count_pos(individual.text),
                    "verbcount": count_verbs(individual.text)
        }

        for ft in features:
            i = ft.get_coordinate_for(individual)
            if i != None:
                b = b + (i,)
            else:
                # this individual is out of range and must be discarded
                individual.distance_to_target = np.inf
                b = None
                break
            
        if b != None:
            individual.coordinate = b
            individual.distance_to_target = us.manhattan(b, goal)
        
        if individual.distance_to_target < 5:
            samples.append(individual) 

    if len(samples) < INITIAL_SEED:
            return samples
    else: 
            return samples[:INITIAL_SEED]


def reseed_individual(seeds):
    # Chooses randomly the seed among the ones that are not covered by the archive
    if len(starting_seeds) > len(seeds):
        seed = random.sample(set(starting_seeds) - seeds, 1)[0]
    else:
        seed = random.choice(starting_seeds)
    sample = creator.Individual(x_test[seed], EXPECTED_LABEL, seed)
    return sample


def evaluate_individual(individual, features, goal, archive):
    if individual.predicted_label == None:
        evaluator.evaluate(individual)
    if DIVERSITY_METRIC == "LATENT" and individual.latent_vector is None:
        individual.compute_latent_vector(doc2vec)
    elif DIVERSITY_METRIC == "HEATMAP" and individual.explanation is None:
        individual.compute_explanation()

    # diversity computation
    individual.sparesness, _ = evaluator.evaluate_sparseness(individual, archive.archive)

    if individual.distance_to_target == None:         
        # original coordinates
        b = tuple()

        individual.features = {
                    "negcount":  count_neg(individual.text),
                    "poscount": count_pos(individual.text),
                    "verbcount": count_verbs(individual.text)
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
    sample = TextMutator(individual).mutate(x_test)
    return sample


def generate_features(meta_file):
    features = []
    with open(meta_file, 'r') as f:
        meta = json.load(f)["features"]
        
    if "NegCount" in FEATURES:
        f3 = Feature("negcount", meta["negcount"]["min"], meta["negcount"]["max"], "count_neg", 25)
        features.append(f3)
    if "PosCount" in FEATURES:
        f2 = Feature("poscount",meta["poscount"]["min"], meta["poscount"]["max"], "count_pos", 25)
        features.append(f2)
    if "VerbCount" in FEATURES:
        f1 = Feature("verbcount",meta["verbcount"]["min"], meta["verbcount"]["max"], "count_verbs", 25)
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



def main():
    # initialize archive
    archive = Archive(TARGET_SIZE)

    features = generate_features(META_FILE)


    # Generate initial population and feature map 
    log.info(f"Generate initial population")  
           

    population = toolbox.population(features, GOAL)

    # Evaluate the individuals
    invalid_ind = [ind for ind in population]

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
            sample = creator.Individual(ind.text, EXPECTED_LABEL, ind.seed)
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
            log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['negcount']}, {individual.features['poscount']}, {individual.features['verbcount']}), performance {individual.ff} and distance {individual.distance_to_target} selected")

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

    

