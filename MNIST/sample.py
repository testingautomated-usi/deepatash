
import json
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import gray2rgb


import rasterization_tools
from explainer import explain_integrated_gradiant, explain_cem

from config import XAI_METHOD
# from lime_mnist import explain_lime
# from lrp_mnist import explain_lrp

class Sample:
    COUNT = 0

    def __init__(self, desc, label, seed):
        self.id = Sample.COUNT
        self.seed = seed
        self.features = {}
        self.xml_desc = desc
        self.purified = rasterization_tools.rasterize_in_memory(self.xml_desc)
        self.expected_label = label
        self.predicted_label = None
        self.confidence = None
        self.ff = None
        self.distance_to_target = np.inf
        self.sparseness = np.inf
        self.coordinate = None
        self.latent_vector = None
        self.explanation = None
        Sample.COUNT += 1

    def to_dict(self):
        return {'id': str(self.id),
                'seed': str(self.seed),
                'expected_label': str(self.expected_label),
                'predicted_label': str(self.predicted_label),
                'misbehaviour': self.is_misbehavior(),
                'performance': str(self.ff),
                'features': self.features,
                'distance to target': str(self.distance_to_target),
                'sparseness': str(self.sparseness)
    }

    def compute_explanation(self):
        if XAI_METHOD == "IG":
            self.explanation = explain_integrated_gradiant(self.purified)
        elif XAI_METHOD == "CEM":
            self.explanation = explain_cem(self.purified)


    def compute_latent_vector(self, encoder):
        mean, _, _ = encoder.predict(self.purified)
        self.latent_vector = mean


    def from_dict(self, the_dict):
        for k in self.__dict__.keys():
            if k in the_dict.keys():
                setattr(self, k, the_dict[k])
        return self

    def dump(self, filename):
        data = self.to_dict()
        filedest = filename+".json"
        with open(filedest, 'w') as f:
            (json.dump(data, f, sort_keys=True, indent=4))

    def save_png(self, filename):
        plt.imsave(filename+'.png', self.purified.reshape(28, 28), cmap='gray', format='png')

    def save_npy(self, filename):
        np.save(filename, self.purified)
        test_img = np.load(filename+'.npy')
        diff = self.purified - test_img
        assert(np.linalg.norm(diff) == 0)

    def save_svg(self, filename):
        data = self.xml_desc
        filedest = filename + ".svg"
        with open(filedest, 'w') as f:
            f.write(data)

    def is_misbehavior(self):
        if str(self.expected_label) == str(self.predicted_label):
            return False
        else:
            return True

    def export(self, dst):
        dst = join(dst, "mbr"+str(self.id))
        self.dump(dst)
        self.save_npy(dst)
        self.save_png(dst)
        self.save_svg(dst)

    def clone(self):
        clone_digit = Sample(self.xml_desc, self.expected_label, self.seed)
        return clone_digit