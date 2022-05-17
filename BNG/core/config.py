from pathlib import Path
import os

class Config:
    GEN_RANDOM = 'GEN_RANDOM'
    GEN_RANDOM_SEEDED = 'GEN_RANDOM_SEEDED'
    GEN_SEQUENTIAL_SEEDED = 'GEN_SEQUENTIAL_SEEDED'
    GEN_DIVERSITY = 'GEN_DIVERSITY'

    EXECTIME = 0
    INVALID = 0

    # mutation operator probability
    MUTATION_EXTENT = 6.0
    MUTPB = 0.7

    def __init__(self):
        # try:
        self.BNG_HOME = f"{str(Path.home())}/Desktop/beamng/trunk" #os.environ['BNG_HOME']
        # except Error:
        #     self.BNG_HOME = f"{str(Path.home())}/Downloads/BeamNG.research.v1.7.0.1"

        print("Setting BNG_HOME to ", self.BNG_HOME)

        # try:
        # self.BNG_USER = os.environ['BNG_USER']
        # except Error:
        self.BNG_USER = f"{str(Path.home())}/Documents/BeamNG.research"

        print("Setting BNG_USER to ", self.BNG_USER)

        self.experiment_name = 'exp'
        self.fitness_weights = (-1.0,)

        self.simulation_save = True
        self.simulation_name = 'beamng_nvidia_runner/sim_$(id)'
        self.keras_model_file = 'self-driving-car-178-2020.h5'
        # self.generator_name = Config.GEN_SEQUENTIAL_SEEDED
        # self.seed_folder = 'population_HQ1'
        self.generator_name = Config.GEN_DIVERSITY
        self.seed_folder = 'initial_pool'
        self.initial_population_folder = "initial_population"
        self.name = ""
        self.seed_folder = 'initial_pool'
        self.initial_population_folder = "initial_population"








