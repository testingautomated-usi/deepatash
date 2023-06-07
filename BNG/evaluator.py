import numpy as np
from utils import get_distance
from config import K
import utils as us


class Evaluator:
    cache = dict()

    def evaluate_mock(self, individual, model_MLP, model_StdSA, model_behaviour):
        nodes = [[item[0], item[1]] for item in individual.ind.m.control_nodes]
        individual.features["mean_lateral_position"] = model_MLP.predict(np.array([np.array(nodes).flatten()]))[0]
        individual.features["sd_steering_angle"] = model_StdSA.predict(np.array([np.array(nodes).flatten()]))[0]
        individual.ind.m.oob_ff = model_behaviour.predict(np.array([np.array(nodes).flatten()]))[0]

    def evaluate_simulation(self, individual, features, goal):
        self.evaluate(individual.ind)            
        individual.features["sd_steering_angle"] = us.sd_steering(individual)
        individual.features["mean_lateral_position"] = us.mean_lateral_position(individual)

        b = tuple()

        for ft in features:
            i = ft.get_coordinate_for(individual)
            if i != None:
                b = b + (i,)
            else:
                # this individual is out of range and must be discarded
                individual.distance_to_target = np.inf
        
        individual.coordinate = b
        if len(b) > 1:
            individual.distance_to_target = us.manhattan(b, goal)
        else:
            individual.distance_to_target = np.inf


    def evaluate(self, ind):
        ind.evaluate()

        border = ind.m.distance_to_boundary
        ind.m.oob_ff = border if border > 0 else -0.1

        return ind.m.oob_ff


    def calculate_dist(self, ind, ind_pop):

        def memoized_dist(ind, ind_pop):
            index_str = tuple(sorted([ind.id, ind_pop.id]))
            if index_str in Evaluator.cache:
                return Evaluator.cache[index_str]
            d = get_distance(ind, ind_pop)
            Evaluator.cache.update({index_str: d})
            return d

        return memoized_dist(ind, ind_pop)


    def dist_from_nearest_archived(self, ind, population, k):
        neighbors = list()
        for ind_pop in population:
            if ind_pop.id != ind.id:
                d = self.calculate_dist(ind, ind_pop)
                if d > 0.0:
                    neighbors.append((d, ind_pop))

        if len(neighbors) == 0:
            assert (len(population) > 0)
            # assert (population[0].id == ind.id)
            return -1.0, ind

        neighbors = sorted(neighbors, key=lambda x: x[0])
        nns = neighbors[:k]
        # k > 1 is not handeled yet
        if k > 1:
            dist = np.mean(nns)
        elif k == 1:
            dist = nns[0][0]
        if dist == 0.0:
            print('bug')
        return dist, nns[0][1]


    def evaluate_sparseness(self, ind, individuals):
        N = (len(individuals))
        # Sparseness is evaluated only if the archive is not empty
        # Otherwise the sparseness is 1
        if (N == 0) or (N == 1 and individuals[0] == ind):
            ind.sparseness = np.inf
            closest_ind = ind
        elif N == 2:
            ind.sparseness, closest_ind = self.dist_from_nearest_archived(ind, individuals, K)
            individuals[0].sparseness = ind.sparseness
            individuals[1].sparseness = ind.sparseness
        else:
            ind.sparseness, closest_ind = self.dist_from_nearest_archived(ind, individuals, K)
        return ind.sparseness, closest_ind