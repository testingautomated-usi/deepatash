# this class must have simple fields in order to be serialized
from core.config import Config



# this class must have simple fields in order to be serialized
class BeamNGConfig(Config):


    SEG_LENGTH = 25
    NUM_SPLINE_NODES =10
    INITIAL_NODE = (0.0, 0.0, -28.0, 8.0)
    ROAD_BBOX_SIZE = (-1000, 0, 1000, 1500)
    EXECTIME = 0
    INVALID = 0

    EVALUATOR_FAKE = 'EVALUATOR_FAKE'
    CONFIDENCE_EVALUATOR = 'CONFIDENCE_EVALUATOR'
    EVALUATOR_LOCAL_BEAMNG = 'EVALUATOR_LOCAL_BEAMNG'
    EVALUATOR_REMOTE_BEAMNG = 'EVALUATOR_REMOTE_BEAMNG'

    def __init__(self):
        super().__init__()

        self.num_control_nodes = 10

        self.MIN_SPEED = 10
        self.MAX_SPEED = 25

        self.beamng_close_at_iteration = False
        self.beamng_evaluator = self.EVALUATOR_LOCAL_BEAMNG
        #self.beamng_evaluator = self.EVALUATOR_FAKE

