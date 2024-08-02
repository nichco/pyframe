import pyframe as pf
import numpy as np
# from dataclasses import dataclass
# from typing import Optional


class CSTube:
    def __init__(self, 
                 radius:np.array,
                 thickness:np.array
                 ):
        
        self.radius = radius
        self.thickness = thickness

    @property
    def area(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**2 - inner_radius**2)
    
    @property
    def ix(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**4 - inner_radius**4) / 2
    
    @property
    def iy(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**4 - inner_radius**4) / 4
    
    @property
    def iz(self):
        inner_radius, outer_radius = self.radius - self.thickness, self.radius
        return np.pi * (outer_radius**4 - inner_radius**4) / 4
    

    def stress(self, element_loads):

        F_x1 = element_loads[:, 0]
        # F_y1 = element_loads[:, 1]
        # F_z1 = element_loads[:, 2]
        M_x1 = element_loads[:, 3]
        M_y1 = element_loads[:, 4]
        M_z1 = element_loads[:, 5]

        F_x2 = element_loads[:, 6]
        # F_y2 = element_loads[:, 7]
        # F_z2 = element_loads[:, 8]
        M_x2 = element_loads[:, 9]
        M_y2 = element_loads[:, 10]
        M_z2 = element_loads[:, 11]


        # average the nodal loads
        F_x = (F_x1 + F_x2) / 2
        # F_y = (F_y1 + F_y2) / 2
        # F_z = (F_z1 + F_z2) / 2
        M_x = (M_x1 + M_x2) / 2
        M_y = (M_y1 + M_y2) / 2
        M_z = (M_z1 + M_z2) / 2

        axial_stress = F_x / self.area
        shear_stress = M_x * self.radius / self.ix

        max_moment = (M_y**2 + M_z**2 + 1E-12)**0.5
        bending_stress = max_moment * self.radius / self.iy

        tensile_stress = axial_stress + bending_stress

        eps = 1E-12
        von_mises_stress = (tensile_stress**2 + 3*shear_stress**2 + eps)**0.5

        
        return von_mises_stress

    


# @dataclass
# class CSTube:
#     radius: float
#     thickness: float

#     @property
#     def type(self):
#         return 'tube'

#     @property
#     def area(self):
#         inner_radius, outer_radius = self.radius - self.thickness, self.radius
#         return np.pi * (outer_radius**2 - inner_radius**2)
    
#     @property
#     def ix(self):
#         inner_radius, outer_radius = self.radius - self.thickness, self.radius
#         return np.pi * (outer_radius**4 - inner_radius**4) / 2

#     @property
#     def iy(self):
#         inner_radius, outer_radius = self.radius - self.thickness, self.radius
#         return np.pi * (outer_radius**4 - inner_radius**4) / 4
    
#     @property
#     def iz(self):
#         inner_radius, outer_radius = self.radius - self.thickness, self.radius
#         return np.pi * (outer_radius**4 - inner_radius**4) / 4