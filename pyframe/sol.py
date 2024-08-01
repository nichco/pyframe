import pyframe as pf
import numpy as np



class Solution:

    def __init__(self, displacement: dict, mesh: dict, stress: dict, 
                 cg: dict, dcg: dict, M: np.array,
                 K: np.array, F: np.array, mass: float):

        self.displacement = displacement
        self.mesh = mesh
        self.stress = stress
        self.cg = cg
        self.dcg = dcg
        self.M = M
        self.K = K
        self.F = F
        self.mass = mass