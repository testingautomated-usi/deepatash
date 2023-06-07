import random
from typing import List

from self_driving.beamng_config import BeamNGConfig
from self_driving.beamng_evaluator import BeamNGEvaluator
from self_driving.beamng_member import BeamNGMember
import logging as log


class BeamNGConfidenceEvaluator(BeamNGEvaluator):
    def __init__(self, config: BeamNGConfig):
        self.config = config

    def evaluate(self, members: List[BeamNGMember]):
        for member in members:
            if not member.distance_to_boundary:
                member.distance_to_boundary = random.uniform(-0.1, 2.0)
                log.debug(f'eval{member}')
