import numpy as np
import pyframe as pf
import csdl_alpha as csdl

class CSDLBeam:

    def __init__(self, name:str, 
                 mesh:csdl.Variable, 
                 material:'pf.Material', 
                 cs:'pf.cs',
                 z=False):
        
        self.name = name
        self.mesh = mesh
        self.material = material
        self.cs = cs
        self.z = z

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
        self.lengths, self.ll, self.mm, self.nn, self.D = self._lengths2(mesh)

    def fix(self, node):

        if node not in self.boundary_conditions:
            self.boundary_conditions.append(node)

    def add_inertial_mass(self, mass, node):
        extra_mass = csdl.Variable(value=np.zeros(self.num_nodes))
        extra_mass = extra_mass.set(csdl.slice[node], mass)
        self.extra_inertial_mass = extra_mass

    def add_load(self, load):
        self.loads = load

    # def _lengths(self, mesh):

    #     lengths = csdl.Variable(value=np.zeros(self.num_elements))
    #     cp = csdl.Variable(value=np.zeros((self.num_elements, 3)))
    #     for i in range(self.num_elements):
    #         diff = mesh[i+1, :] - mesh[i, :]
    #         L = csdl.norm(diff)
    #         lengths = lengths.set(csdl.slice[i], L)
    #         cp = cp.set(csdl.slice[i, :], diff / L)

    #     # precomps for transforms
    #     ll = cp[:, 0]
    #     mm = cp[:, 1]
    #     nn = cp[:, 2]
    #     D = (ll**2 + mm**2)**0.5

    #     return lengths, ll, mm, nn, D
    
    def _lengths2(self, mesh):
        # Compute the squared differences
        diffs = mesh[1:] - mesh[:-1]
        # Sum the squared differences along the rows and take the square root
        lengths = csdl.norm(diffs, axes=(1,))
        exl = csdl.expand(lengths, (self.num_elements, 3), action='i->ij')
        cp = diffs / exl

        # precomps for transforms
        ll = cp[:, 0]
        mm = cp[:, 1]
        nn = cp[:, 2]
        D = (ll**2 + mm**2)**0.5


        return lengths, ll, mm, nn, D

        
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
        nAEL = -AEL
        GJL = G*J/L
        nGJL = -GJL

        EIz = E*Iz
        EIzL = EIz/L
        EIzL2 = EIzL/L
        EIzL3 = EIzL2/L
        EIzL312 = 12*EIzL3
        nEIzL312 = -EIzL312
        EIzL26 = 6*EIzL2
        nEIzL26 = -EIzL26
        EIzL4 = 4*EIzL
        EIzL2 = 2*EIzL

        EIy = E*Iy
        EIyL = EIy/L
        EIyL2 = EIyL/L
        EIyL3 = EIyL2/L
        EIyL26 = 6*EIyL2
        nEIyL26 = -EIyL26
        EIyL312 = 12*EIyL3
        nEIyL312 = -EIyL312
        EIyL4 = 4*EIyL
        EIyL2 = 2*EIyL

        local_stiffness = local_stiffness.set(csdl.slice[:, 0, 0], AEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 1], EIzL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 5], EIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 1], EIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 2], EIyL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 4], nEIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 2], nEIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 3, 3], GJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 4], EIyL4)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 5], EIzL4)

        local_stiffness = local_stiffness.set(csdl.slice[:, 0, 6], nAEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 7], nEIzL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 1, 11], EIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 8], nEIyL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 2, 10], nEIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 3, 9], nGJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 8], EIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 4, 10], EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 7], nEIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 5, 11], EIzL2)

        local_stiffness = local_stiffness.set(csdl.slice[:, 6, 0], nAEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 1], nEIzL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 5], nEIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 2], nEIyL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 4], EIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 9, 3], nGJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 2], nEIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 4], EIyL2)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 1], EIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 5], EIzL2)

        local_stiffness = local_stiffness.set(csdl.slice[:, 6, 6], AEL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 7], EIzL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 7, 11], nEIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 7], nEIzL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 8], EIyL312)
        local_stiffness = local_stiffness.set(csdl.slice[:, 8, 10], EIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 8], EIyL26)
        local_stiffness = local_stiffness.set(csdl.slice[:, 9, 9], GJL)
        local_stiffness = local_stiffness.set(csdl.slice[:, 10, 10], EIyL4)
        local_stiffness = local_stiffness.set(csdl.slice[:, 11, 11], EIzL4)

        self.local_stiffness_bookshelf = local_stiffness

        return local_stiffness
    
    def _local_mass_matrices(self):
        
        A = self.cs.area
        rho = self.material.density
        J = self.cs.ix
        L = self.lengths

        # coefficients
        aa = L / 2
        aa2 = aa**2
        coef = rho * A * aa / 105
        coef70 = coef * 70
        coef78 = coef * 78
        coef35 = coef * 35
        ncoef35 = -coef35
        coef27 = coef * 27
        coef22aa = coef * 22 * aa
        ncoef22aa = -coef22aa
        coef13aa = coef * 13 * aa
        ncoef13aa = -coef13aa
        coef8aa2 = coef * 8 * aa2
        ncoef6aa2 = -coef * 6 * aa2
        rx2 = J / A
        ncoef35rx2 = ncoef35 * rx2

        local_mass = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        local_mass = local_mass.set(csdl.slice[:, 0, 0], coef70)
        local_mass = local_mass.set(csdl.slice[:, 1, 1], coef78)
        local_mass = local_mass.set(csdl.slice[:, 2, 2], coef78)
        local_mass = local_mass.set(csdl.slice[:, 3, 3], coef78 * rx2)
        local_mass = local_mass.set(csdl.slice[:, 2, 4], ncoef22aa)
        local_mass = local_mass.set(csdl.slice[:, 4, 2], ncoef22aa)
        local_mass = local_mass.set(csdl.slice[:, 4, 4], coef8aa2)
        local_mass = local_mass.set(csdl.slice[:, 1, 5], coef22aa)
        local_mass = local_mass.set(csdl.slice[:, 5, 1], coef22aa)
        local_mass = local_mass.set(csdl.slice[:, 5, 5], coef8aa2)
        local_mass = local_mass.set(csdl.slice[:, 0, 6], coef35)
        local_mass = local_mass.set(csdl.slice[:, 6, 0], coef35)
        local_mass = local_mass.set(csdl.slice[:, 6, 6], coef70)
        local_mass = local_mass.set(csdl.slice[:, 1, 7], coef27)
        local_mass = local_mass.set(csdl.slice[:, 7, 1], coef27)
        local_mass = local_mass.set(csdl.slice[:, 5, 7], coef13aa)
        local_mass = local_mass.set(csdl.slice[:, 7, 5], coef13aa)
        local_mass = local_mass.set(csdl.slice[:, 7, 7], coef78)
        local_mass = local_mass.set(csdl.slice[:, 2, 8], coef27)
        local_mass = local_mass.set(csdl.slice[:, 8, 2], coef27)
        local_mass = local_mass.set(csdl.slice[:, 4, 8], ncoef13aa)
        local_mass = local_mass.set(csdl.slice[:, 8, 4], ncoef13aa)
        local_mass = local_mass.set(csdl.slice[:, 8, 8], coef78)
        local_mass = local_mass.set(csdl.slice[:, 3, 9], ncoef35rx2)
        local_mass = local_mass.set(csdl.slice[:, 9, 3], ncoef35rx2)
        local_mass = local_mass.set(csdl.slice[:, 9, 9], coef70 * rx2)
        local_mass = local_mass.set(csdl.slice[:, 2, 10], coef13aa)
        local_mass = local_mass.set(csdl.slice[:, 10, 2], coef13aa)
        local_mass = local_mass.set(csdl.slice[:, 4, 10], ncoef6aa2)
        local_mass = local_mass.set(csdl.slice[:, 10, 4], ncoef6aa2)
        local_mass = local_mass.set(csdl.slice[:, 8, 10], coef22aa)
        local_mass = local_mass.set(csdl.slice[:, 10, 8], coef22aa)
        local_mass = local_mass.set(csdl.slice[:, 10, 10], coef8aa2)
        local_mass = local_mass.set(csdl.slice[:, 1, 11], ncoef13aa)
        local_mass = local_mass.set(csdl.slice[:, 11, 1], ncoef13aa)
        local_mass = local_mass.set(csdl.slice[:, 5, 11], ncoef6aa2)
        local_mass = local_mass.set(csdl.slice[:, 11, 5], ncoef6aa2)
        local_mass = local_mass.set(csdl.slice[:, 7, 11], ncoef22aa)
        local_mass = local_mass.set(csdl.slice[:, 11, 7], ncoef22aa)
        local_mass = local_mass.set(csdl.slice[:, 11, 11], coef8aa2)


        return local_mass


    def _transforms(self):

        T = csdl.Variable(value=np.zeros((self.num_elements, 12, 12)))

        block = csdl.Variable(value=np.zeros((self.num_elements, 3, 3)))
        for i in range(self.num_elements):
            ll = self.ll[i]
            mm = self.mm[i]
            nn = self.nn[i]
            nmm = -mm # precomp for speed
            D = self.D[i]
            nmmD = nmm / D
            llD = ll / D

            if self.z:
                block = block.set(csdl.slice[i, 0, 2], 1)
                block = block.set(csdl.slice[i, 1, 1], 1)
                block = block.set(csdl.slice[i, 2, 0], -1)
            else:
                block = block.set(csdl.slice[i, 0, 0], ll)
                block = block.set(csdl.slice[i, 0, 1], mm)
                block = block.set(csdl.slice[i, 0, 2], nn)
                block = block.set(csdl.slice[i, 1, 0], nmmD)
                block = block.set(csdl.slice[i, 1, 1], llD)
                block = block.set(csdl.slice[i, 2, 0], -nn * llD)
                block = block.set(csdl.slice[i, 2, 1], nn * nmmD)
                block = block.set(csdl.slice[i, 2, 2], D)

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
        displacements = csdl.Variable(value=np.zeros((self.num_elements, 12)))
        lsb = self.local_stiffness_bookshelf
        tb = self.transformations_bookshelf

        for i in range(self.num_elements):
            idxa, idxb = map[i], map[i+1]
            displacements = displacements.set(csdl.slice[i, 0:6], U[idxa:idxa+6])
            displacements = displacements.set(csdl.slice[i, 6:12], U[idxb:idxb+6])

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

