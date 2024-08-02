import numpy as np
import pyframe as pf
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import csdl_alpha as csdl

class CSDLFrame:
    def __init__(self):

        self.beams = []
        self.joints = []
        self.acc = None

    def add_beam(self, beam:'pf.CSDLBeam'):

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

        helper = {node: i for i, node in enumerate(nodes)}

        for beam in self.beams:
            map = beam.map
            
            for i in range(beam.num_nodes):
                map[i] = helper[map[i]] * 6

        return dim, num


    def solve(self):

        dim, num = self._utils()
        
        
        # create the global stiffness matrix
        # and the global mass matrix
        K = csdl.Variable(value=np.zeros((dim, dim)))
        M = csdl.Variable(value=np.zeros((dim, dim)))

        for beam in self.beams:
            transforms = beam._transforms()
            # lists of stiffness matrices for each element in the global frame
            transformed_stiffness_matrices = beam._transform_stiffness_matrices(transforms)
            transformed_mass_matrices = beam._transform_mass_matrices(transforms)
            # add the elemental stiffness/mass matrices to their locations in the 
            # global stiffness/mass matrix
            map = beam.map

            for i in range(beam.num_elements):
                stiffness = transformed_stiffness_matrices[i]
                mass_matrix = transformed_mass_matrices[i]
                idxa, idxb = map[i], map[i+1]

                K = K.set(csdl.slice[idxa:idxa+6, idxa:idxa+6], K[idxa:idxa+6, idxa:idxa+6] + stiffness[:6, :6])
                K = K.set(csdl.slice[idxa:idxa+6, idxb:idxb+6], K[idxa:idxa+6, idxb:idxb+6] + stiffness[:6, 6:])
                K = K.set(csdl.slice[idxb:idxb+6, idxa:idxa+6], K[idxb:idxb+6, idxa:idxa+6] + stiffness[6:, :6])
                K = K.set(csdl.slice[idxb:idxb+6, idxb:idxb+6], K[idxb:idxb+6, idxb:idxb+6] + stiffness[6:, 6:])

                M = M.set(csdl.slice[idxa:idxa+6, idxa:idxa+6], M[idxa:idxa+6, idxa:idxa+6] + mass_matrix[:6, :6])
                M = M.set(csdl.slice[idxa:idxa+6, idxb:idxb+6], M[idxa:idxa+6, idxb:idxb+6] + mass_matrix[:6, 6:])
                M = M.set(csdl.slice[idxb:idxb+6, idxa:idxa+6], M[idxb:idxb+6, idxa:idxa+6] + mass_matrix[6:, :6])
                M = M.set(csdl.slice[idxb:idxb+6, idxb:idxb+6], M[idxb:idxb+6, idxb:idxb+6] + mass_matrix[6:, 6:])


        # # assemble te global loads vector
        F = csdl.Variable(value=np.zeros((dim)))
        for beam in self.beams:
            loads = beam.loads
            map = beam.map

            for i in range(beam.num_nodes):
                idx = map[i]
                F = F.set(csdl.slice[idx:idx+6], F[idx:idx+6] + loads[i, :])

        
        # add any inertial loads
        acc = self.acc
        if acc is not None:
            # expanded_acc = np.tile(self.acc, num)
            expanded_acc = csdl.expand(acc, (num, 6), action='i->ji').flatten()
            primary_inertial_loads = csdl.matvec(M, expanded_acc)
            F += primary_inertial_loads

            # added inertial masses are resolved as loads
            for beam in self.beams:
                extra_mass = beam.extra_inertial_mass
                extra_inertial_loads = csdl.outer(extra_mass, acc)
                map = beam.map

                for i in range(beam.num_nodes):
                    idx = map[i]
                    F = F.set(csdl.slice[idx:idx+6], F[idx:idx+6] + extra_inertial_loads[i, :])




        # apply boundary conditions
        for beam in self.beams:
            map = beam.map
            for node in beam.boundary_conditions:
                idx = map[node]

                for i in range(6):
                    # if dof[i] == 1:
                        # zero the row/column then put a 1 in the diagonal
                        K = K.set(csdl.slice[idx + i, :], 0)
                        K = K.set(csdl.slice[:, idx + i], 0)
                        K = K.set(csdl.slice[idx + i, idx + i], 1)

                        # zero the corresponding load index as well
                        F = F.set(csdl.slice[idx + i], 0)



        # solve the system of equations
        U = csdl.solve_linear(K, F)
        # K = sp.csr_matrix(K)
        # U = spla.spsolve(K, F)


        # find the displacements
        displacement = {}
        for beam in self.beams:
            displacement[beam.name] = csdl.Variable(value=np.zeros((beam.num_nodes, 3)))
            map = beam.map

            for i in range(beam.num_nodes):
                idx = map[i]
                # extract the (x, y, z) nodal displacement
                displacement[beam.name] = displacement[beam.name].set(csdl.slice[i, :], U[idx:idx+3])


        # calculate the elemental loads and stresses
        stress = {}
        for beam in self.beams:
            # elemental loads
            element_loads = beam._recover_loads(U)
            element_loads = csdl.vstack(element_loads)
            # perform a stress recovery
            beam_stress = beam.cs.stress(element_loads)

            stress[beam.name] = beam_stress


        # mass properties
        mass = 0
        for beam in self.beams:
            mass += beam._mass()



        


        return pf.Solution(displacement=displacement,
                           stress=stress,
                           M=M,
                           K=K,
                           F=F,
                           mass=mass,)





    