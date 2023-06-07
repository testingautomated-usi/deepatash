from datetime import datetime
import json
import sys
import random
from deap import base, creator, tools
import logging as log
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from utils import time_to_sec


# local
from core.config import Config
import self_driving.beamng_config as cfg
import self_driving.beamng_individual as BeamNGIndividual
from archive import Archive
from config import POPSIZE, to_json
import self_driving.beamng_problem as BeamNGProblem
from config import GOAL, FEATURES, POPSIZE, RESEEDUPPERBOUND,\
     RUN_TIME, DIVERSITY_METRIC, META_FILE, TARGET_SIZE, INITIAL_POP, INITIAL_SEED, RUN_ID, APPROACH
import utils as us
from sample import Sample
from evaluator import Evaluator
from feature import Feature

evaluator = Evaluator()

# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# Define the individual.
creator.create("Individual", Sample, fitness=creator.FitnessMin)

lr_model_MLP = LinearRegression(fit_intercept=False)
lr_model_StdSA = LinearRegression(fit_intercept=False)
lr_model_behaviour = LinearRegression(fit_intercept=False)

# for training LR
X = []
Y1 = []
Y2 = []
Y3 = []

def train_lrs(X, Y1, Y2, Y3):
    log.info("LRs are trained")
    lr_model_MLP.fit(X, Y1)
    lr_model_StdSA.fit(X, Y2)
    lr_model_behaviour.fit(X, Y3)


def evaluate_lrs(_X, y1_test, y2_test, y3_test):
    y1_prediction = lr_model_MLP.predict(np.array(_X))
    y2_prediction = lr_model_StdSA.predict(np.array(_X))
    y3_prediction = lr_model_behaviour.predict(np.array(_X))

    from sklearn.metrics import mean_squared_error
    log.info(f"MSE lr MLP: {mean_squared_error(np.array(y1_test),y1_prediction)}")
    log.info(f"MSE lr StdSA: {mean_squared_error(np.array(y2_test),y2_prediction)}")
    log.info(f"MSE lr behaviour: {mean_squared_error(np.array(y3_test),y3_prediction)}")


def generate_initial_pop(problem, features, goal):

    samples = []
    for id in range(0, INITIAL_SEED):
        max_angle = random.randint(10,100)
        road = problem.generate_random_member(max_angle)
        ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
        ind.seed = id
        sample = creator.Individual(ind)
        samples.append(sample)    

        evaluator.evaluate(sample.ind)

        sample.features = {
                    "segment_count": us.segment_count(sample),
                    "sd_steering_angle": us.sd_steering(sample),
                    "mean_lateral_position": us.mean_lateral_position(sample),
                    "curvature": us.curvature(sample) 
        }

        b = tuple()
    
        for ft in features:
            i = ft.get_coordinate_for(sample)
            if i != None:
                b = b + (i,)
            else:
                # this individual is out of range and must be discarded
                sample.distance_to_target = np.inf
        
        sample.coordinate = b
        sample.distance_to_target = us.manhattan(b, goal)


        # for training a regressor
        nodes = [[item[0], item[1]] for item in sample.ind.m.control_nodes]
        X.append(np.array(nodes).flatten())
        Y1.append(sample.features["mean_lateral_position"])
        Y2.append(sample.features["sd_steering_angle"])
        Y3.append(sample.ind.m.oob_ff)
        log.info(f"ind {sample.id} with seed {sample.seed} and ({sample.features['segment_count']}, {sample.features['curvature']}, {sample.features['sd_steering_angle']}, {sample.features['mean_lateral_position']}), performance {sample.ind.m.oob_ff} and distance {sample.distance_to_target} evaluated")

    initial_pop = sorted(samples, key=lambda x: x.distance_to_target, reverse=False)[:INITIAL_POP]

    return initial_pop 


def reseed_individual(problem, seed):
    max_angle = random.randint(10,100)
    road = problem.generate_random_member(max_angle)
    ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
    ind.seed = seed
    sample = creator.Individual(ind)
    sample.ind.m.oob_ff = None
    return sample


def evaluate_individual(individual, features, goal, archive, mock):
    
    if mock == False:
        if individual.ind.m.oob_ff == None or individual.ind.m.distance_to_boundary == None or individual.features == {}:
            evaluator.evaluate(individual.ind)
    else:
        evaluator.evaluate_mock(individual, lr_model_MLP, lr_model_StdSA, lr_model_behaviour)

    # diversity computation
    individual.sparseness, _ = evaluator.evaluate_sparseness(individual, archive.archive)

    if individual.distance_to_target == None:         
        # original coordinates
        b = tuple()

        if mock == False:            
            individual.features = {
                "segment_count": us.segment_count(individual),
                "curvature": us.curvature(individual), 
                "sd_steering_angle" : us.sd_steering(individual),
                "mean_lateral_position" : us.mean_lateral_position(individual)
            }
        else:
            individual.features["segment_count"] = us.segment_count(individual)
            individual.features["curvature"] = us.curvature(individual)

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


    if mock == False:
        nodes = [[item[0], item[1]] for item in individual.ind.m.control_nodes]
        X.append(np.array(nodes).flatten())
        Y1.append(individual.features["mean_lateral_position"])
        Y2.append(individual.features["sd_steering_angle"])
        Y3.append(individual.ind.m.oob_ff)
    
    log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['segment_count']}, {individual.features['curvature']}, {individual.features['sd_steering_angle']}, {individual.features['mean_lateral_position']}), performance {individual.ind.m.oob_ff} and distance {individual.distance_to_target} evaluated")
    
    return (individual.distance_to_target,)


def mutate_individual(sample):
    sample.ind.mutate()
    return sample


def generate_features(meta_file):

    features = []
    with open(meta_file, 'r') as f:
        meta = json.load(f)["features"]
        
    if "Curvature" in FEATURES:
        f1 = Feature("curvature",meta["curvature"]["min"], meta["curvature"]["max"], "curvature", 25)
        features.append(f1)
    if "SegmentCount" in FEATURES:
        f2 = Feature("segment_count",meta["segment_count"]["min"], meta["segment_count"]["max"], "segment_count", int(meta["segment_count"]["max"])+1)
        features.append(f2)
    if "MeanLateralPosition" in FEATURES:
        f3 = Feature("mean_lateral_position", meta["mean_lateral_position"]["min"], meta["mean_lateral_position"]["max"], "mean_lateral_position", 25)
        features.append(f3)
    if "SDSteeringAngle" in FEATURES:
        f4 = Feature("sd_steering_angle",meta["sd_steering_angle"]["min"], meta["sd_steering_angle"]["max"], "sd_steering", 25)
        features.append(f4)
    return features


toolbox.register("population", generate_initial_pop)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selBest)
toolbox.register("mutate", mutate_individual)


def main(name):
    # initialize archive
    archive = Archive(TARGET_SIZE)

    last_seed_index = POPSIZE
    _config = cfg.BeamNGConfig()
    _config.name = name
    problem = BeamNGProblem.BeamNGProblem(_config)

    start_time = datetime.now()

    features = generate_features(META_FILE)


    # Generate initial population and feature map 
    log.info(f"Generate initial population")  
           

    population = toolbox.population(problem, features, GOAL)

    fitnesses = [i.distance_to_target for i in population]

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit,)

    # Select the next generation population
    population = toolbox.select(population, POPSIZE)


    # Begin the generational process
    gen = 1

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    mock = False
    train_lr = False

    while time_to_sec(elapsed_time) < RUN_TIME:

        if time_to_sec(elapsed_time) > 0.2 * RUN_TIME:
            mock = True
            if train_lr == False:
                train_lrs(X, Y1, Y2, Y3)
                train_lr = True


        log.info(f"Iteration: {gen}")

        # Vary the population.
        offspring = []
        for ind in population:
            new_ind = ind.ind.clone()
            sample = creator.Individual(new_ind)
            sample.ind.seed = ind.id
            offspring.append(sample)
                
        # Mutation.
        log.info("Mutation")
        for ind in offspring:
            mut = toolbox.mutate(ind)
            ind.ind.m.oob_ff = None
            if mut == None:
                ind = reseed_individual(problem, last_seed_index)
                last_seed_index += 1

        # Reseeding
        if len(archive.archive) > 0:
            log.info(f"Reseeding")
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            candidate_seeds = archive.archived_seeds
            log.info(f"Archive seeds: {candidate_seeds}")

            for i in range(seed_range):
                ind = population[len(population) - i - 1]
                new_ind = reseed_individual(problem, last_seed_index)
                population[len(population) - i - 1] = new_ind
                log.info(f"ind {ind.id} with seed {ind.seed} reseeded by {new_ind.id} with seed {new_ind.seed}")
                last_seed_index += 1
                
            for i in range(len(population)):
                if population[i].seed in archive.archived_seeds:
                    ind = population[i]
                    new_ind = reseed_individual(problem, last_seed_index)
                    population[i] = new_ind
                    log.info(f"ind {ind.id} with seed {ind.seed} reseeded by {new_ind.id} with seed {new_ind.seed}")
                    last_seed_index += 1


        # Evaluate the individuals
        invalid_ind = [ind for ind in offspring + population]
        fitnesses = [toolbox.evaluate(i, features, GOAL, archive, mock) for i in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        # Update archive
        for ind in offspring + population:
            is_added = archive.update_archive(ind, evaluator, GOAL, features, mock)
            # if a new ind added to archive, recompute all the sparsenesses
            if is_added:
                archive.recompute_sparseness_archive(evaluator)

        # Select the next generation population
        population = toolbox.select(population + offspring, POPSIZE)

        for individual in population:
            log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['segment_count']}, {individual.features['curvature']}, {individual.features['sd_steering_angle']}, {individual.features['mean_lateral_position']}), performance {individual.ind.m.oob_ff} and distance {individual.distance_to_target} selected")

        gen += 1
        log.info(f"Archive: {len(archive.archive)}")
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time


        log.info("Elapsed time: " + str(elapsed_time))
        log.info("EXECTIME: " + str(Config.EXECTIME))
        log.info(f"Archive: {len(archive.archive)}")

    archive.export_archive(name)
    if len(archive.archive) > 0:
        x1 = []
        y1_test = []
        y2_test = []
        y3_test = []
        for ind in archive.archive:
            nodes = [[item[0], item[1]] for item in individual.ind.m.control_nodes]
            x1.append(np.array(nodes).flatten())
            y1_test.append(individual.features["mean_lateral_position"])
            y2_test.append(individual.features["sd_steering_angle"])
            y3_test.append(individual.ind.m.oob_ff)
        
        evaluate_lrs(x1, y1_test, y2_test, y3_test)

    return population

    

if __name__ == "__main__": 

    start_time = datetime.now()
    # run = sys.argv[1]
    name = f"logs/{RUN_ID}-{APPROACH}_-features_{FEATURES[0]}-{FEATURES[1]}-diversity_{DIVERSITY_METRIC}"
    
    Path(name).mkdir(parents=True, exist_ok=True)

    to_json(name)
    log_to = f"{name}/logs.txt"

    # Setup logging
    us.setup_logging(log_to)
    print("Logging results to " + log_to)

    pop = main(name)
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    log.info("elapsed_time:"+ str(elapsed_time))

    print("GAME OVER")

    

    
