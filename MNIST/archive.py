

from os import makedirs
from os.path import exists
import logging as log

class Archive:

    def __init__(self, _target_size):
        self.archive = list()
        self.archived_seeds = set()
        self.tshd_members = dict()
        self.target_size = _target_size

    def get_archive(self):
        return self.archive


    def update_archive(self, ind, evaluator):
        flag = False
        if ind not in self.archive and ind.distance_to_target <= 1:
            # archive is empty
            if len(self.archive) == 0:
                log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['moves']}, {ind.features['orientation']}, {ind.features['bitmaps']}), performance {ind.ff}, sparseness {ind.sparseness} and distance {ind.distance_to_target} added to archive")
                self.archive.append(ind)
                self.archived_seeds.add(ind.seed)
                flag = False
            else:
                # Find the member of the archive that is closest to the candidate.
                d_min, _ = evaluator.evaluate_sparseness(ind, self.archive)
                # archive is not full
                if len(self.archive)/self.target_size < 1:
                    # not the same sparseness
                    if d_min > 0:                    
                        log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['moves']}, {ind.features['orientation']}, {ind.features['bitmaps']}), performance {ind.ff}, sparseness {ind.sparseness} and distance {ind.distance_to_target} added to archive")
                        self.archive.append(ind)
                        self.archived_seeds.add(ind.seed)
                        flag = True
                
                # archive is full
                else:
                    # find the farthest individual distance to target (worst) and smallest sparseness
                    c = sorted(self.archive, key=lambda x: (x.distance_to_target, -x.sparseness), reverse=True)[0]
                    if c.distance_to_target > ind.distance_to_target:
                        log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['moves']}, {ind.features['orientation']}, {ind.features['bitmaps']}), performance {ind.ff}, sparseness {ind.sparseness} and distance {ind.distance_to_target} added to archive")
                        log.info(f"ind {c.id} with seed {c.seed} and ({c.features['moves']}, {c.features['orientation']}, {c.features['bitmaps']}), performance {c.ff}, sparseness {c.sparseness} and distance {c.distance_to_target} removed from archive")
                        self.archive.remove(c)
                        self.archive.append(ind)
                        self.archived_seeds.add(ind.seed)
                        flag = True
                    elif c.distance_to_target == ind.distance_to_target:
                        # ind has better performance
                        if ind.ff < c.ff:
                            log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['moves']}, {ind.features['orientation']}, {ind.features['bitmaps']}), performance {ind.ff}, sparseness {ind.sparseness} and distance {ind.distance_to_target} added to archive")
                            log.info(f"ind {c.id} with seed {c.seed} and ({c.features['moves']}, {c.features['orientation']}, {c.features['bitmaps']}), performance {c.ff}, sparseness {c.sparseness} and distance {c.distance_to_target} removed from archive")
                            self.archive.remove(c)                                
                            self.archive.append(ind)
                            self.archived_seeds.add(ind.seed)
                            flag = True
                        # c and ind have the same performance
                        elif ind.ff == c.ff:
                            # ind has better sparseness                        
                            if d_min > c.sparseness:
                                log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['moves']}, {ind.features['orientation']}, {ind.features['bitmaps']}), performance {ind.ff}, sparseness {ind.sparseness} and distance {ind.distance_to_target} added to archive")
                                log.info(f"ind {c.id} with seed {c.seed} and ({c.features['moves']}, {c.features['orientation']}, {c.features['bitmaps']}), performance {c.ff}, sparseness {c.sparseness} and distance {c.distance_to_target} removed from archive")
                                self.archive.remove(c)                            
                                self.archive.append(ind)
                                self.archived_seeds.add(ind.seed)
                                flag = True
                                
                        


        return flag


    def recompute_sparseness_archive(self, evaluator):
        self.archived_seeds = set()
        for ind in self.archive:
            self.archived_seeds.add(ind.seed)
            ind.sparseness, _ = evaluator.evaluate_sparseness(ind, self.archive)

    def export_archive(self, dst):
        if not exists(dst):
            makedirs(dst)
        for ind in self.archive:
            print(".", end='', flush=True)
            ind.export(dst)