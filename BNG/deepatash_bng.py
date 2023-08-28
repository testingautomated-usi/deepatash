import os
from config import APPROACH

if APPROACH == "ga":
    os.system("python ga_method_lr.py")
elif APPROACH == "nsga2":
    os.system("python nsga2_method_lr.py")