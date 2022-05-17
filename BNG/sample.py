
import json
import numpy as np
from pathlib import Path
from self_driving.beamng_road_imagery import BeamNGRoadImagery
from self_driving.beamng_individual import BeamNGIndividual


class Sample:
    COUNT = 0
    # At which radius we interpret a tuns as a straight?
    # MAX_MIN_RADIUS = 200
    MAX_MIN_RADIUS = 170

    def __init__(self, ind):
        self.ind: BeamNGIndividual = ind 
        self.id = Sample.COUNT
        self.seed = ind.seed
        self.features = {}
        self.distance_to_target = np.inf
        self.sparseness = np.inf
        self.misbehaviour = None
        Sample.COUNT += 1


    def to_dict(self):
        return {
                'id': self.id,
                'elapsed': str(self.ind.m.elapsed),
                'timestamp': str(self.ind.m.timestamp),
                'misbehaviour': self.is_misbehaviour(),
                'performance': str(self.ind.m.distance_to_boundary),
                'features': self.features,
                'distance to target': str(self.distance_to_target),
                'sparseness': str(self.sparseness),
                'control_nodes': self.ind.m.control_nodes,
                'sample_nodes': self.ind.m.sample_nodes,
                'num_spline_nodes': self.ind.m.num_spline_nodes
                # 'road_bbox_size': self.ind.m.road_bbox.bbox.bounds
        }

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))


    def clone(self):
        res = Sample(self.ind)
        return res


    def evaluate(self):
        return self.ind.evaluate()


    def is_misbehaviour(self):
        if self.ind.m.distance_to_boundary < 0:
            return True
        else:
            return False


    def export(self, name):
        sim_folder = Path(f"{name}/sim_{self.id}")
        sim_folder.mkdir(parents=True, exist_ok=True)

        destination_sim_json = f"{sim_folder}\\simulation.full.json"
        destination_road = f"{sim_folder}\\road.json"


        with open(destination_sim_json, 'w') as f:
            f.write(json.dumps({
                self.ind.m.simulation.f_params: self.ind.m.simulation.params._asdict(),
                self.ind.m.simulation.f_info: self.ind.m.simulation.info.__dict__,
                self.ind.m.simulation.f_road: self.ind.m.simulation.road.to_dict(),
                self.ind.m.simulation.f_records: [r._asdict() for r in self.ind.m.simulation.states]
            }))


        with open(destination_road, 'w') as f:
            road = self.to_dict()
            f.write(json.dumps(road))

        road_imagery = BeamNGRoadImagery.from_sample_nodes(self.ind.m.sample_nodes)
        image_path = sim_folder.joinpath(f"img_{self.ind.m.name}")
        road_imagery.save(image_path.with_suffix('.jpg'))
        road_imagery.save(image_path.with_suffix('.svg'))