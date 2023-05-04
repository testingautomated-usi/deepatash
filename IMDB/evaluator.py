import numpy as np
from utils import get_distance
from config import K
from predictor import Predictor

class Evaluator:
    cache = dict()

    def evaluate(self, ind):
        ind.ff = None          
        ind.predicted_label, ind.confidence = Predictor.predict(ind.text)

        # Calculate fitness function
        ind.ff = ind.confidence if ind.confidence > 0 else -0.1
            
        return ind.ff


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