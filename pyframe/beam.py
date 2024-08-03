import numpy as np
import pyframe as pf


class Beam:

    def __init__(self, name:str, 
                 mesh:np.array, 
                 material:'pf.Material', 
                 cs:'pf.cs'):
        
        self.name = name
        self.mesh = mesh
        self.material = material
        self.cs = cs

        self.num_nodes = mesh.shape[0]
        self.num_elements = self.num_nodes - 1

        self.loads = np.zeros((self.num_nodes, 6))
        self.extra_inertial_mass = np.zeros((self.num_nodes))

        self.boundary_conditions = []

        # map the beam nodes to the global indices
        self.map = []

        # storage
        self.local_stiffness_bookshelf = None
        self.transformations_bookshelf = None

        # precompute lengths
        self.lengths = self._lengths(mesh)

    def fix(self, node):

        if node not in self.boundary_conditions:
            self.boundary_conditions.append(node)

    def add_inertial_mass(self, mass, node):
        inertial_mass = np.zeros(self.num_nodes)
        inertial_mass[node] = mass
        self.extra_inertial_mass += inertial_mass

    def add_load(self, load):
        self.loads += load

    # def _lengths(self):

    #     lengths = np.zeros(self.num_elements)
    #     for i in range(self.num_elements):
    #         lengths[i] = np.linalg.norm(self.mesh[i+1] - self.mesh[i])

    #     return lengths
    
    def _lengths(self, mesh):
        # Compute the squared differences
        squared_diffs = (mesh[1:] - mesh[:-1]) ** 2
        # Sum the squared differences along the rows and take the square root
        lengths = np.sqrt(np.sum(squared_diffs, axis=1))
        return lengths

    # def _lengths(self):
    #     lengths = [np.linalg.norm(self.mesh[i+1] - self.mesh[i]) for i in range(self.num_elements)]
    #     return np.array(lengths)

        
    def _local_stiffness_matrices(self):

        A = self.cs.area
        E, G = self.material.E, self.material.G
        Iz = self.cs.iz
        Iy = self.cs.iy
        J = self.cs.ix
        L = self.lengths

        local_stiffness = np.zeros((self.num_elements, 12, 12))

        # pre-computations for speed
        EIy = E*Iy
        EIz = E*Iz
        EIzL3 = EIz/L**3
        EIzL2 = EIz/L**2
        EIyL3 = EIy/L**3
        EIyL2 = EIy/L**2

        local_stiffness[:, 0, 0] = A*E/L
        local_stiffness[:, 1, 1] = 12*EIzL3
        local_stiffness[:, 1, 5] = 6*EIzL2
        local_stiffness[:, 5, 1] = 6*EIzL2
        local_stiffness[:, 2, 2] = 12*EIyL3
        local_stiffness[:, 2, 4] = -6*EIyL2
        local_stiffness[:, 4, 2] = -6*EIyL2
        local_stiffness[:, 3, 3] = G*J/L
        local_stiffness[:, 4, 4] = 4*EIy/L
        local_stiffness[:, 5, 5] = 4*EIz/L

        local_stiffness[:, 0, 6] = -A*E/L
        local_stiffness[:, 1, 7] = -12*EIzL3
        local_stiffness[:, 1, 11] = 6*EIzL2
        local_stiffness[:, 2, 8] = -12*EIyL3
        local_stiffness[:, 2, 10] = -6*EIyL2
        local_stiffness[:, 3, 9] = -G*J/L
        local_stiffness[:, 4, 8] = 6*EIyL2
        local_stiffness[:, 4, 10] = 2*EIy/L
        local_stiffness[:, 5, 7] = -6*EIzL2
        local_stiffness[:, 5, 11] = 2*EIz/L

        local_stiffness[:, 6, 0] = -A*E/L
        local_stiffness[:, 7, 1] = -12*EIzL3
        local_stiffness[:, 7, 5] = -6*EIzL2
        local_stiffness[:, 8, 2] = -12*EIyL3
        local_stiffness[:, 8, 4] = 6*EIyL2
        local_stiffness[:, 9, 3] = -G*J/L
        local_stiffness[:, 10, 2] = -6*EIyL2
        local_stiffness[:, 10, 4] = 2*EIy/L
        local_stiffness[:, 11, 1] = 6*EIzL2
        local_stiffness[:, 11, 5] = 2*EIz/L

        local_stiffness[:, 6, 6] = A*E/L
        local_stiffness[:, 7, 7] = 12*EIzL3
        local_stiffness[:, 7, 11] = -6*EIzL2
        local_stiffness[:, 11, 7] = -6*EIzL2
        local_stiffness[:, 8, 8] = 12*EIyL3
        local_stiffness[:, 8, 10] = 6*EIyL2
        local_stiffness[:, 10, 8] = 6*EIyL2
        local_stiffness[:, 9, 9] = G*J/L
        local_stiffness[:, 10, 10] = 4*EIy/L
        local_stiffness[:, 11, 11] = 4*EIz/L

        self.local_stiffness_bookshelf = local_stiffness

        return local_stiffness
    
    def _local_mass_matrices(self):
        
        A = self.cs.area
        rho = self.material.density
        J = self.cs.ix
        L = self.lengths

        # coefficients
        aa = L / 2
        coef = rho * A * aa / 105
        rx2 = J / A

        local_mass = np.zeros((self.num_elements, 12, 12))

        local_mass[:, 0, 0] = coef * 70
        local_mass[:, 1, 1] = coef * 78
        local_mass[:, 2, 2] = coef * 78
        local_mass[:, 3, 3] = coef * 78 * rx2
        local_mass[:, 2, 4] = coef * -22 * aa
        local_mass[:, 4, 2] = coef * -22 * aa
        local_mass[:, 4, 4] = coef * 8 * aa**2
        local_mass[:, 1, 5] = coef * 22 * aa
        local_mass[:, 5, 1] = coef * 22 * aa
        local_mass[:, 5, 5] = coef * 8 * aa**2
        local_mass[:, 0, 6] = coef * 35
        local_mass[:, 6, 0] = coef * 35
        local_mass[:, 6, 6] = coef * 70
        local_mass[:, 1, 7] = coef * 27
        local_mass[:, 7, 1] = coef * 27
        local_mass[:, 5, 7] = coef * 13 * aa
        local_mass[:, 7, 5] = coef * 13 * aa
        local_mass[:, 7, 7] = coef * 78
        local_mass[:, 2, 8] = coef * 27
        local_mass[:, 8, 2] = coef * 27
        local_mass[:, 4, 8] = coef * -13 * aa
        local_mass[:, 8, 4] = coef * -13 * aa
        local_mass[:, 8, 8] = coef * 78
        local_mass[:, 3, 9] = coef * -35 * rx2
        local_mass[:, 9, 3] = coef * -35 * rx2
        local_mass[:, 9, 9] = coef * 70 * rx2
        local_mass[:, 2, 10] = coef * 13 * aa
        local_mass[:, 10, 2] = coef * 13 * aa
        local_mass[:, 4, 10] = coef * -6 * aa**2
        local_mass[:, 10, 4] = coef * -6 * aa**2
        local_mass[:, 8, 10] = coef * 22 * aa
        local_mass[:, 10, 8] = coef * 22 * aa
        local_mass[:, 10, 10] = coef * 8 * aa**2
        local_mass[:, 1, 11] = coef * -13 * aa
        local_mass[:, 11, 1] = coef * -13 * aa
        local_mass[:, 5, 11] = coef * -6 * aa**2
        local_mass[:, 11, 5] = coef * -6 * aa**2
        local_mass[:, 7, 11] = coef * -22 * aa
        local_mass[:, 11, 7] = coef * -22 * aa
        local_mass[:, 11, 11] = coef * 8 * aa**2


        return local_mass


    def _transforms(self):

        transforms = []
        
        lengths = self.lengths
        T = np.zeros((12, 12))
        mesh = self.mesh

        for i in range(self.num_elements):
            cp = (mesh[i + 1, :] - mesh[i, :]) / lengths[i]
            ll, mm, nn = cp[0], cp[1], cp[2]
            D = (ll**2 + mm**2)**0.5

            block = np.zeros((3, 3))
            if D == 0:
                block[0, 2] = 1
                block[1, 1] = 1
                block[2, 0] = -1
            else:
                block[0, 0] = ll
                block[0, 1] = mm
                block[0, 2] = nn
                block[1, 0] = -mm / D
                block[1, 1] = ll / D
                block[2, 0] = -ll * nn / D
                block[2, 1] = -mm * nn / D
                block[2, 2] = D

            T[0:3, 0:3] = block
            T[3:6, 3:6] = block
            T[6:9, 6:9] = block
            T[9:12, 9:12] = block

            transforms.append(T)

        self.transformations_bookshelf = transforms

        return transforms

    def _transform_stiffness_matrices(self, transforms):
        
        local_stiffness_matrices = self._local_stiffness_matrices()

        # transformed_stiffness_matrices = []

        # for i in range(self.num_elements):
        #     T = transforms[i]
        #     local_stiffness = local_stiffness_matrices[i, :, :]
        #     TKT = T.T @ local_stiffness @ T
        #     transformed_stiffness_matrices.append(TKT)

        # Shape: (num_elements, n, n)
        T_transpose = np.einsum('ijk->ikj', transforms)
        # Shape: (num_elements, n, n)
        T_transpose_K = np.einsum('ijk,ikl->ijl', T_transpose, local_stiffness_matrices)
        # Shape: (num_elements, n, n)
        transformed_stiffness_matrices = np.einsum('ijk,ikl->ijl', T_transpose_K, transforms)

        return transformed_stiffness_matrices
    

    def _transform_mass_matrices(self, transforms):

        local_mass_matrices = self._local_mass_matrices()

        # transformed_mass_matrices = []

        # for i in range(self.num_elements):
        #     T = transforms[i]
        #     local_mass = local_mass_matrices[i, :, :]
        #     TMT = T.T @ local_mass @ T
        #     transformed_mass_matrices.append(TMT)

        # Shape: (num_elements, n, n)
        T_transpose = np.einsum('ijk->ikj', transforms)
        # Shape: (num_elements, n, n)
        T_transpose_M = np.einsum('ijk,ikl->ijl', T_transpose, local_mass_matrices)
        # Shape: (num_elements, n, n)
        transformed_mass_matrices = np.einsum('ijk,ikl->ijl', T_transpose_M, transforms)

        return transformed_mass_matrices
    

    def _recover_loads(self, U):

        map = self.map
        # loads = []
        # displacement = np.zeros((12))
        displacements = np.zeros((self.num_elements, 12))
        lsb = self.local_stiffness_bookshelf
        tb = self.transformations_bookshelf

        for i in range(self.num_elements):
            idxa, idxb = map[i], map[i+1]
            # displacement[0:6] = U[idxa:idxa+6]
            # displacement[6:12] = U[idxb:idxb+6]
            displacements[i, 0:6] = U[idxa:idxa+6]
            displacements[i, 6:12] = U[idxb:idxb+6]

            # local_stiffness = lsb[i, :, :] # matrix
            # T = tb[i] # list of matrices

            # loads.append(local_stiffness @ T @ displacement)

        transformed_displacements = np.einsum('ijk,ik->ij', tb, displacements)

        # Compute loads
        loads = np.einsum('ijk,ik->ij', lsb, transformed_displacements)

        return loads
    

    def _mass(self):

        lengths = self.lengths
        rho = self.material.density
        area = self.cs.area

        element_masses = area * lengths * rho
        beam_mass = np.sum(element_masses)

        return beam_mass

