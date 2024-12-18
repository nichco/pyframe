import numpy as np
import pyframe as pf
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh


class Frame:
    def __init__(self):

        self.beams = []
        self.joints = []
        self.acc = None
        self.K = None
        self.M = None

        self.K_no_bc = None
        self.M_no_bc = None

        self.dim = None
        self.displacement = {}
        self.stress = None
        self.U = None

        self.u0 = None
        self.F = None


    def add_beam(self, beam:'pf.Beam'):

        self.beams.append(beam)


    def add_joint(self, members:list, nodes:list):
        
        self.joints.append({'members': members, 'nodes': nodes})


    def add_acc(self, acc:np.array):

        if acc.shape != (6,):
            raise ValueError("acc must have shape 6")

        if self.acc is None:
            self.acc = acc
        else:
            raise ValueError("acc is not None")
        
    
    def _utils(self):

        idx = 0
        for beam in self.beams:
            beam.map.extend(range(idx, idx + beam.num_nodes))
            idx += beam.num_nodes

        # re-assign joint nodes
        for joint in self.joints:
            members = joint['members']
            nodes = joint['nodes']
            index = members[0].map[nodes[0]]

            for i, member in enumerate(members):
                if i != 0:
                    member.map[nodes[i]] = index

        # nodes = set()
        # for beam in self.beams:
        #     for i in range(beam.num_nodes):
        #         nodes.add(beam.map[i])
        # a faster version of the above code
        nodes = {node for beam in self.beams for node in beam.map[:beam.num_nodes]}
        
        # the global dimension is the number of unique nodes times
        # the degrees of freedom per node
        num = len(nodes)
        dim = num * 6
        self.dim = dim

        helper = {node: i for i, node in enumerate(nodes)}

        for beam in self.beams:
            map = beam.map
            
            for i in range(beam.num_nodes):
                map[i] = helper[map[i]] * 6

        return dim, num
    
    def compute_mass(self):

        # mass properties
        mass = 0
        for beam in self.beams:
            mass += beam._mass()

        return mass
    

    def compute_stress(self):

        # calculate the elemental loads and stresses
        stress = {}
        for beam in self.beams:
            # elemental loads
            element_loads = beam._recover_loads(self.U)
            element_loads = np.vstack(element_loads)
            # perform a stress recovery
            beam_stress = beam.cs.stress(element_loads)
            stress[beam.name] = beam_stress

        self.stress = stress
        return None
    
    # for dynamic simulation
    def beam_stress(self, beam, u):

        # calculate the elemental loads and stresses
        # stress = {}
        # for beam in self.beams:
        # elemental loads
        element_loads = beam._recover_loads(u)
        element_loads = np.vstack(element_loads)
        # perform a stress recovery
        beam_stress = beam.cs.stress(element_loads)

        return beam_stress
    

    def compute_natural_frequency(self):
        K = self.K
        M = self.M

        w, v = eigh(K, M)
        omega = np.sqrt(w)

        sorted_indices = np.argsort(omega)
        natural_frequencies_sorted = omega[sorted_indices]
        mode_shapes_sorted = v[:, sorted_indices]

        shapes = [] # a list of dictionaries
        for i in range(mode_shapes_sorted.shape[1]):
            mode = mode_shapes_sorted[:, i]
            mode /= np.linalg.norm(mode)
            # shape is a dictionary of displacements
            shape = self.compute_displacements(mode)
            shapes.append(shape)

        return natural_frequencies_sorted, shapes
    

    def compute_displacements(self):
        
        for beam in self.beams:
            self.displacement[beam.name] = np.empty((beam.num_nodes, 3))
            map = beam.map

            for i in range(beam.num_nodes):
                idx = map[i]
                # extract the (x, y, z) nodal displacement
                self.displacement[beam.name][i, :] = self.U[idx:idx+3]

        return None


    def solve(self):

        dim, num = self._utils()
        
        # create the global stiffness matrix
        # and the global mass matrix
        K = np.zeros((dim, dim))
        M = np.zeros((dim, dim))

        for beam in self.beams:
            # lists of stiffness matrices for each element in the global frame
            transformed_stiffness_matrices = beam.transformed_stiffness
            transformed_mass_matrices = beam.transformed_mass
            # add the elemental stiffness/mass matrices to their locations in the 
            # global stiffness/mass matrix
            map = beam.map

            for i in range(beam.num_elements):
                stiffness = transformed_stiffness_matrices[i]
                mass_matrix = transformed_mass_matrices[i]
                idxa, idxb = map[i], map[i+1]

                K[idxa:idxa+6, idxa:idxa+6] += stiffness[:6, :6]
                K[idxa:idxa+6, idxb:idxb+6] += stiffness[:6, 6:]
                K[idxb:idxb+6, idxa:idxa+6] += stiffness[6:, :6]
                K[idxb:idxb+6, idxb:idxb+6] += stiffness[6:, 6:]

                M[idxa:idxa+6, idxa:idxa+6] += mass_matrix[:6, :6]
                M[idxa:idxa+6, idxb:idxb+6] += mass_matrix[:6, 6:]
                M[idxb:idxb+6, idxa:idxa+6] += mass_matrix[6:, :6]
                M[idxb:idxb+6, idxb:idxb+6] += mass_matrix[6:, 6:]

        self.K_no_bc = K
        self.M_no_bc = M

        # # assemble the global loads vector
        F = np.zeros((dim))
        for beam in self.beams:
            loads = beam.loads
            map = beam.map

            for i in range(beam.num_nodes):
                idx = map[i]
                F[idx:idx+6] += loads[i, :]

        
        # add any inertial loads
        if self.acc is not None:
            expanded_acc = np.tile(self.acc, num)
            primary_inertial_loads = M @ expanded_acc
            F += primary_inertial_loads

            # added inertial masses are resolved as loads
            for beam in self.beams:
                extra_mass = beam.extra_inertial_mass
                extra_inertial_loads = np.outer(extra_mass, self.acc)
                map = beam.map

                for i in range(beam.num_nodes):
                    idx = map[i]
                    F[idx:idx+6] += extra_inertial_loads[i, :]




        # apply boundary conditions
        indices = []
        for beam in self.beams:
            map = beam.map
            for node in beam.boundary_conditions:
                idx = map[node]

                for i in range(6):
                    # if dof[i] == 1:
                    indices.append(idx + i)

        # zero the row/column then put a 1 in the diagonal
        K[indices, :] = 0
        K[:, indices] = 0
        K[indices, indices] = 1
        # zero the corresponding load index as well
        F[indices] = 0

        M[indices, :] = 0
        M[:, indices] = 0
        M[indices, indices] = 1


        self.K = K
        self.M = M
        self.F = F

        # initial conditions for dynamic sim
        no_loads = np.zeros((dim))
        # u0 = np.linalg.solve(M, no_loads)
        u0 = np.linalg.solve(K, no_loads)
        self.u0 = u0


        # solve the system of equations
        # U = np.linalg.solve(K, F)
        K = sp.csr_matrix(K)
        U = spla.spsolve(K, F)
        self.U = U



        


        self.compute_displacements()
        self.compute_stress()


        return None





