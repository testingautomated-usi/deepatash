
from config import ARCHIVE_THRESHOLD, TARGET_THRESHOLD
from os import makedirs
from os.path import exists
from evaluator import Evaluator
import numpy as np
from utils import get_distance
import logging as log

class Archive:

    def __init__(self):
        self.archive = list()
        self.archived_seeds = set()
        self.tshd_members = dict()

    def get_archive(self):
        return self.archive

    # def update_archive(self, ind):
    #     if ind not in self.archive:
    #         bucket = [arc_ind for arc_ind in self.archive if arc_ind.seed == ind.seed]
    #         if len(bucket) == 0:
    #             if ind.distance_to_target <= 1 and ind.is_misbehavior() == True:
    #                 self.archive.append(ind)
    #                 self.archived_seeds.add(ind.seed)
    #                 self.tshd_members[ind.seed] = ind
    #         else:
    #             # Find the member of the archive that is closest to the candidate.
    #             d_min = np.inf
    #             i = 0
    #             while i < len(bucket):
    #                 distance_archived = get_distance(ind, bucket[i])
    #                 if distance_archived < d_min:
    #                     d_min = distance_archived
    #                 i += 1
    #             # Decide whether to add the candidate to the archive
    #             # Verify whether the candidate is close to the existing member of the archive
    #             # Note: 'close' is defined according to a user-defined threshold
    #             if d_min > ARCHIVE_THRESHOLD:
    #                 if len(bucket) < MAX_BUCKET_SIZE and ind.is_misbehavior() == True:
    #                     self.archive.append(ind)
    #                 elif len(bucket) == MAX_BUCKET_SIZE:
    #                     bucket.sort(key=lambda x: x.ff)
    #                     tshd = bucket[-1]
    #                     if ind.ff < tshd.ff and ind.distance_to_target <= 1:
    #                         self.tshd_members[ind.seed] = ind
    #                         self.archive.remove(tshd)
    #                         self.archive.append(ind)


    def update_archive(self, ind, evaluator):
        if ind not in self.archive:
            if len(self.archive) == 0:
                if ind.distance_to_target <= TARGET_THRESHOLD and ind.is_misbehaviour() == True:
                    log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['SegmentCount']}, {ind.features['Curvature']}, {ind.features['SDSteeringAngle']}),{ind.features['MeanLateralPosition']} and distance {ind.distance_to_target} added to archive")

                    self.archive.append(ind)
                    self.archived_seeds.add(ind.seed)
            else:
                # Find the member of the archive that is closest to the candidate.
                d_min, closest_ind =  evaluator.evaluate_sparseness(ind, self.archive)
                # Decide whether to add the candidate to the archive
                # Verify whether the candidate is close to the existing member of the archive
                # Note: 'close' is defined according to a user-defined threshold
                if d_min > ARCHIVE_THRESHOLD:
                    if ind.distance_to_target <= TARGET_THRESHOLD and ind.is_misbehaviour() == True:
                        log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['SegmentCount']}, {ind.features['Curvature']}, {ind.features['SDSteeringAngle']}),{ind.features['MeanLateralPosition']} and distance {ind.distance_to_target} added to archive")
                        self.archive.append(ind)
                        self.archived_seeds.add(ind.seed)
                else:
                    if closest_ind.distance_to_target > ind.distance_to_target:
                        log.info(f"ind {ind.id} with seed {ind.seed} and ({ind.features['SegmentCount']}, {ind.features['Curvature']}, {ind.features['SDSteeringAngle']}),{ind.features['MeanLateralPosition']} and distance {ind.distance_to_target} added to archive")
                        log.info(f"ind {closest_ind.id} with seed {closest_ind.seed} and ({closest_ind.features['SegmentCount']}, {closest_ind.features['Curvature']}, {closest_ind.features['SDSteeringAngle']}),{closest_ind.features['MeanLateralPosition']} and distance {closest_ind.distance_to_target} removed archive")

                        self.archive.append(ind)
                        self.archived_seeds.add(ind.seed)
                        self.archive.remove(closest_ind)

    def export_archive(self, dst):
        if not exists(dst):
            makedirs(dst)
        for ind in self.archive:
            print(".", end='', flush=True)
            ind.export(dst)