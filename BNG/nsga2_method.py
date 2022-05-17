from datetime import datetime
from pickle import POP
import sys
import random
from this import d
from deap import base, creator
from deap.tools.emo import selNSGA2
from evaluator import Evaluator
import logging as log
from pathlib import Path
import numpy as np

# local
from core.config import Config
import self_driving.beamng_config as cfg
import self_driving.beamng_individual as BeamNGIndividual
from archive import Archive
from config import POPSIZE, to_json
import self_driving.beamng_problem as BeamNGProblem
from config import GOAL, FEATURES, POPSIZE, RESEEDUPPERBOUND, RUN_TIME, DIVERSITY_METRIC
import utils as us
from sample import Sample

evaluator = Evaluator()

# DEAP framework setup.
toolbox = base.Toolbox()
# Define a bi-objective fitness function.
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, 1))
# Define the individual.
creator.create("Individual", Sample, fitness=creator.FitnessMulti)



def generate_initial_pop(problem):
    samples = []
    initialpop = 1
    for i in range(1, initialpop+1):
        max_angle = random.randint(10,100)
        road = problem.generate_random_member(max_angle)
        ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
        ind.seed = i
        sample = creator.Individual(ind)
        samples.append(sample)        
    return samples 

def reseed_individual(problem, seed):
    max_angle = random.randint(10,100)
    road = problem.generate_random_member(max_angle)
    ind = BeamNGIndividual.BeamNGIndividual(road, problem.config)
    ind.oob_ff = None
    ind.seed = seed
    sample = creator.Individual(ind)
    return sample

def evaluate_individual(individual, features, goal, archive):

    if individual.ind.oob_ff == None:
        evaluator.evaluate(individual.ind)

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
                    "MinRadius": us.new_min_radius(individual),
                    "SegmentCount": us.segment_count(individual),
                    "DirCoverage": us.direction_coverage(individual),
                    "SDSteeringAngle": us.sd_steering(individual),
                    "MeanLateralPosition": us.mean_lateral_position(individual),
                    "Curvature": us.curvature(individual) 
        }
        
        individual.coordinate = b
        individual.distance_to_target = us.manhattan(a, goal)
    
    log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['SegmentCount']}, {individual.features['Curvature']}, {individual.features['SDSteeringAngle']}, {individual.features['MeanLateralPosition']}) and distance {individual.distance_to_target} evaluated")

    return individual.ind.oob_ff, individual.distance_to_target, individual.sparseness

def mutate_individual(individual):
    sample = individual.clone()
    sample.ind.mutate()
    return sample

def generate_features():
    features = []

    if "MeanLateralPosition" in FEATURES:
        f3 = ["MeanLateralPosition", 2, "mean_lateral_position"]
        features.append(f3)
    if "SegmentCount" in FEATURES:
        f2 = ["SegmentCount", 1, "segment_count"]
        features.append(f2)
    if "Curvature" in FEATURES:
        f1 = ["Curvature", 1, "curvature"]
        features.append(f1)
    if "SDSteeringAngle" in FEATURES:
        f0 = ["SDSteeringAngle", 7, "sd_steering"]
        features.append(f0)
    
    return features
       
def goal_acheived(population):
    for sample in population:
        if sample.distance_to_target == 0:
            print("Goal achieved")
            return True
    return False

toolbox.register("population", generate_initial_pop)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", selNSGA2)
toolbox.register("mutate", mutate_individual)


def main(name):
    last_seed_index = POPSIZE
    _config = cfg.BeamNGConfig()
    _config.name = name
    problem = BeamNGProblem.BeamNGProblem(_config)

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
           

    population = toolbox.population(problem)

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

    h_3 = True
    h_5 = True

    while condition:
        log.info(f"Iteration: {gen}")

        # Vary the population.
        # offspring = [toolbox.clone(ind) for ind in population]
        offspring = []
        for ind in population:
            sample = creator.Individual(ind.ind)
            sample.ind.oob_ff = None
            offspring.append(sample)
                
        # Mutation.
        log.info("Mutation")
        for ind in offspring:
            toolbox.mutate(ind)

        # Reseeding
        if len(archive.archive) > 0:
            log.info(f"Reseeding")
            seed_range = random.randrange(1, RESEEDUPPERBOUND)
            # candidate_seeds = archive.archived_seeds

            for i in range(seed_range):
                population[len(population) - i - 1] = reseed_individual(problem, last_seed_index)
                last_seed_index += 1

            # for i in range(len(population)):
            #      if population[i].seed in archive.archived_seeds:
            #          population[i] = reseed_individual(problem)

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
            log.info(f"ind {individual.id} with seed {individual.seed} and ({individual.features['SegmentCount']}, {individual.features['Curvature']}, {individual.features['SDSteeringAngle']}, {individual.features['MeanLateralPosition']}) and distance {individual.distance_to_target} selected")


        gen += 1

        end_time = datetime.now()
        elapsed_time = end_time - start_time


        if Config.EXECTIME > 10800 and h_3:
            archive.export_archive(name+"/3h")
            h_3 = False
        elif Config.EXECTIME > 18000 and h_5:
            archive.export_archive(name+"/5h")
            h_5 = False
        if Config.EXECTIME > RUN_TIME: #gen == GEN: #or goal_acheived(population):
            condition = False

        log.info("Elapsed time: " + str(elapsed_time))
        log.info("EXECTIME: " + str(Config.EXECTIME))
        log.info(f"Archive: {len(archive.archive)}")

    archive.export_archive(name+"/10h")
    return population


if __name__ == "__main__": 

    start_time = datetime.now()
    run = sys.argv[1]
    name = f"logs/{run}-nsga2_-features_{FEATURES[0]}-{FEATURES[1]}-diversity_{DIVERSITY_METRIC}"
    
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

    

    
