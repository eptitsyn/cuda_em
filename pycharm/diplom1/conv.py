import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts
import math
import pickle
import numbers
import mpl_toolkits.mplot3d.art3d as art3d
import json

with open("Output.txt", "r") as text_file:
    deser = text_file.read()

deserialized_a = pickle.loads(deser)

pickle.dump(deserialized_a, open("Output22b.txt", "wb"))

#    with open("Output.txt", "w") as text_file:
#        text_file.write(serialized)
