import pyframe as pf
import numpy as np



class Solution:

    def __init__(self, displacement: dict, stress: dict, 
                 M: np.array,
                 K: np.array, F: np.array, mass: float):

        self.displacement = displacement
        self.stress = stress
        self.M = M
        self.K = K
        self.F = F
        self.mass = mass