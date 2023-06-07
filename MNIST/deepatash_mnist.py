import os
from config import APPROACH

if APPROACH == "ga":
    os.system("python ga_method.py")
elif APPROACH == "nsga2":
    os.system("python nsga2_method.py")