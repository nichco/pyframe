import numpy as np
import pyframe as pf
import csdl_alpha as csdl

class CSDLBeam:

    def __init__(self, name:str, 
                 mesh:csdl.Variable, 
                 material:'pf.Material', 
                 cs:'pf.cs'):
        
        self.name = name
        self.mesh = mesh
        self.material = material
        self.cs = cs

        self.num_nodes = mesh.shape[0]
        self.num_elements = self.num_nodes - 1

        # self.loads = csdl.Variable(value=np.zeros((self.num_nodes, 6)))
        # self.extra_inertial_mass = csdl.Variable(value=np.zeros((self.num_nodes)))
        self.loads = None
        self.extra_inertial_mass = None

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
        extra_mass = csdl.Variable(value=np.zeros(self.num_nodes))
        extra_mass = extra_mass.set(csdl.slice[node], mass)
        self.extra_inertial_mass = extra_mass

    def add_load(self, load):
        self.loads = load

    def _lengths(self, mesh):

        lengths = csdl.Variable(value=np.zeros(self.num_elements))
        for i in range(self.num_elements):
            lengths = lengths.set(csdl.slice[i], csdl.norm(mesh[i+1] - mesh[i]))

        return lengths
    
    # def _lengths(self, mesh):
    #     # Compute the squared differences
    #     squared_diffs = (mesh[1:] - mesh[:-1])**2
    #     # Sum the squared differences along the rows and take the square root
    #     lengths = (csdl.sum(squared_diffs, axes=1))**0.5
    #     return lengths

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

        local_stiffness = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        # pre-computations for speed
        AEL = A*E/L
        GJL = G*J/L
        EIy = E*Iy
        EIz = E*Iz
        EIzL3 = EIz/L**3
        EIzL2 = EIz/L**2
        EIzL = EIz/L
        EIyL3 = EIy/L**3
        EIyL2 = EIy/L**2
        EIyL = EIy/L

        local_stiffness = local_stiffness.set(csdl.slice[:, 0, 0], AEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 1], 12*EIzL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 5], 6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 1], 6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 2], 12*EIyL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 4], -6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 2], -6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 3, 3], GJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 4], 4*EIyL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 5], 4*EIzL)

        local_stiffness = local_stiffness.set(csdl.slice[:, 0, 6], -AEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 7], -12*EIzL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 11], 6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 8], -12*EIyL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 10], -6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 3, 9], -GJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 8], 6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 10], 2*EIyL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 7], -6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 11], 2*EIzL)

        local_stiffness = local_stiffness.set(csdl.slice[:, 6, 0], -AEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 1], -12*EIzL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 5], -6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 2], -12*EIyL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 4], 6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 9, 3], -GJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 2], -6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 4], 2*EIyL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 1], 6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 5], 2*EIzL)

        local_stiffness = local_stiffness.set(csdl.slice[:, 6, 6], AEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 7], 12*EIzL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 11], -6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 7], -6*EIzL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 8], 12*EIyL3)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 10], 6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 8], 6*EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 9, 9], GJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 10], 4*EIyL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 11], 4*EIzL)

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

        local_mass = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        local_mass = local_mass.set(csdl.slice[:, 0, 0], coef * 70)
        local_mass = local_mass.set(csdl.slice[:, 1, 1], coef * 78)
        local_mass = local_mass.set(csdl.slice[:, 2, 2], coef * 78)
        local_mass = local_mass.set(csdl.slice[:, 3, 3], coef * 78 * rx2)
        local_mass = local_mass.set(csdl.slice[:, 2, 4], coef * -22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 4, 2], coef * -22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 4, 4], coef * 8 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 1, 5], coef * 22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 5, 1], coef * 22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 5, 5], coef * 8 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 0, 6], coef * 35)
        local_mass = local_mass.set(csdl.slice[:, 6, 0], coef * 35)
        local_mass = local_mass.set(csdl.slice[:, 6, 6], coef * 70)
        local_mass = local_mass.set(csdl.slice[:, 1, 7], coef * 27)
        local_mass = local_mass.set(csdl.slice[:, 7, 1], coef * 27)
        local_mass = local_mass.set(csdl.slice[:, 5, 7], coef * 13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 7, 5], coef * 13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 7, 7], coef * 78)
        local_mass = local_mass.set(csdl.slice[:, 2, 8], coef * 27)
        local_mass = local_mass.set(csdl.slice[:, 8, 2], coef * 27)
        local_mass = local_mass.set(csdl.slice[:, 4, 8], coef * -13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 8, 4], coef * -13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 8, 8], coef * 78)
        local_mass = local_mass.set(csdl.slice[:, 3, 9], coef * -35 * rx2)
        local_mass = local_mass.set(csdl.slice[:, 9, 3], coef * -35 * rx2)
        local_mass = local_mass.set(csdl.slice[:, 9, 9], coef * 70 * rx2)
        local_mass = local_mass.set(csdl.slice[:, 2, 10], coef * 13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 10, 2], coef * 13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 4, 10], coef * -6 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 10, 4], coef * -6 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 8, 10], coef * 22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 10, 8], coef * 22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 10, 10], coef * 8 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 1, 11], coef * -13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 11, 1], coef * -13 * aa)
        local_mass = local_mass.set(csdl.slice[:, 5, 11], coef * -6 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 11, 5], coef * -6 * aa**2)
        local_mass = local_mass.set(csdl.slice[:, 7, 11], coef * -22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 11, 7], coef * -22 * aa)
        local_mass = local_mass.set(csdl.slice[:, 11, 11], coef * 8 * aa**2)


        return local_mass


    def _transforms(self):

        # transforms = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))
        
        lengths = self.lengths
        # T = csdl.Variable(value=np.zeros((12, 12)))
        T = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))
        mesh = self.mesh

        block = csdl.Variable(value=np.zeros((self.num_elements, 3, 3)))
        for i in range(self.num_elements):
            cp = (mesh[i + 1, :] - mesh[i, :]) / lengths[i]
            ll, mm, nn = cp[0], cp[1], cp[2]
            D = (ll**2 + mm**2)**0.5

            # block = csdl.Variable(value=np.zeros((3, 3)))
            block = block.set(csdl.slice[i, 0, 0], ll)
            block = block.set(csdl.slice[i, 0, 1], mm)
            block = block.set(csdl.slice[i, 0, 2], nn)
            block = block.set(csdl.slice[i, 1, 0], -mm / D)
            block = block.set(csdl.slice[i, 1, 1], ll / D)
            block = block.set(csdl.slice[i, 2, 0], -ll * nn / D)
            block = block.set(csdl.slice[i, 2, 1], -mm * nn / D)
            block = block.set(csdl.slice[i, 2, 2], D)

            # T = T.set(csdl.slice[i, 0:3, 0:3], block)
            # T = T.set(csdl.slice[i, 3:6, 3:6], block)
            # T = T.set(csdl.slice[i, 6:9, 6:9], block)
            # T = T.set(csdl.slice[i, 9:12, 9:12], block)

            # transforms = transforms.set(csdl.slice[i, :, :], T)

        T = T.set(csdl.slice[:, 0:3, 0:3], block)
        T = T.set(csdl.slice[:, 3:6, 3:6], block)
        T = T.set(csdl.slice[:, 6:9, 6:9], block)
        T = T.set(csdl.slice[:, 9:12, 9:12], block)

        # self.transformations_bookshelf = transforms
        self.transformations_bookshelf = T

        return T

    def _transform_stiffness_matrices(self, transforms):
        
        local_stiffness_matrices = self._local_stiffness_matrices()

        # transformed_stiffness_matrices = []

        # for i in range(self.num_elements):
        #     T = transforms[i]
        #     local_stiffness = local_stiffness_matrices[i, :, :]
        #     TKT = csdl.matmat(csdl.transpose(T), csdl.matmat(local_stiffness, T))
        #     transformed_stiffness_matrices.append(TKT)

        # Shape: (num_elements, n, n)
        T_transpose = csdl.einsum(transforms, action='ijk->ikj')
        # Shape: (num_elements, n, n)
        T_transpose_K = csdl.einsum(T_transpose, local_stiffness_matrices, action='ijk,ikl->ijl')
        # Shape: (num_elements, n, n)
        transformed_stiffness_matrices = csdl.einsum(T_transpose_K, transforms, action='ijk,ikl->ijl')


        return transformed_stiffness_matrices
    

    def _transform_mass_matrices(self, transforms):

        local_mass_matrices = self._local_mass_matrices()

        # transformed_mass_matrices = []

        # for i in range(self.num_elements):
        #     T = transforms[i]
        #     local_mass = local_mass_matrices[i, :, :]
        #     TMT = csdl.matmat(csdl.transpose(T), csdl.matmat(local_mass, T))
        #     transformed_mass_matrices.append(TMT)

        # Shape: (num_elements, n, n)
        T_transpose = csdl.einsum(transforms, action='ijk->ikj')
        # Shape: (num_elements, n, n)
        T_transpose_M = csdl.einsum(T_transpose, local_mass_matrices, action='ijk,ikl->ijl')
        # Shape: (num_elements, n, n)
        transformed_mass_matrices = csdl.einsum(T_transpose_M, transforms, action='ijk,ikl->ijl')

        return transformed_mass_matrices
    

    def _recover_loads(self, U):

        map = self.map
        # loads = []
        # displacement = csdl.Variable(value=np.zeros((12)))
        displacements = csdl.Variable(value=np.zeros((self.num_elements, 12)))
        lsb = self.local_stiffness_bookshelf
        tb = self.transformations_bookshelf

        for i in range(self.num_elements):
            idxa, idxb = map[i], map[i+1]
            # displacement = displacement.set(csdl.slice[0:6], U[idxa:idxa+6])
            # displacement = displacement.set(csdl.slice[6:12], U[idxb:idxb+6])
            displacements = displacements.set(csdl.slice[i, 0:6], U[idxa:idxa+6])
            displacements = displacements.set(csdl.slice[i, 6:12], U[idxb:idxb+6])

            # local_stiffness = lsb[i, :, :] # matrix
            # T = tb[i] # list of matrices
            # loads.append(csdl.matvec(local_stiffness, csdl.matvec(T, displacement)))

        # local_stiffness = lsb[:, :, :]
        # transformations = np.array(tb)

        # Perform transformations
        transformed_displacements = csdl.einsum(tb, displacements, action='ijk,ik->ij')

        # Compute loads
        loads = csdl.einsum(lsb, transformed_displacements, action='ijk,ik->ij')

        return loads
    

    def _mass(self):

        lengths = self.lengths
        rho = self.material.density
        area = self.cs.area

        element_masses = area * lengths * rho
        beam_mass = csdl.sum(element_masses)

        return beam_mass

