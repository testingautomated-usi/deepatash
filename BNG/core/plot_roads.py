import sys
import os
from pathlib import Path
from datetime import datetime
#sys.path.insert(0, r'C:\DeepHyperion-BNG')
#sys.path.append(os.path.dirname(os.path.dirname(os.path.join(__file__))))
path = Path(os.path.abspath(__file__))
# This corresponds to DeepHyperion-BNG
sys.path.append(str(path.parent))
sys.path.append(str(path.parent.parent))
from core.plot_utils import plot_roads

path = sys.argv[1]

# The immediate file path
directory_contents = os.listdir(path)
abs_path = os.path.abspath(path)
for item in directory_contents:
# Filter for directories
    if os.path.isdir(f"{abs_path}/{item}"):
        names = os.listdir(f"{abs_path}/{item}")
        for name in names:
            if os.path.isdir(f"{abs_path}/{item}/{name}"):
                features = name.split("_")
                plot_roads(f"{abs_path}/{item}", features[0], features[1])
