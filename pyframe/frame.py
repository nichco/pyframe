import numpy as np
import pyframe as pf



class Frame:
    def __init__(self):

        self.beams = []
        self.joints = []
        self.acc = None

    def add_beam(self, beam:'pf.Beam'):

        self.beams.append(beam)

    def add_joint(self, members:list, nodes:list):
        
        self.joints.append({'members': members, 'nodes': nodes})

    def add_acc(self, acc:np.array):

        if self.acc is None:
            self.acc = acc
        else:
            raise ValueError("acc is not None")


    def solve(self):

        # fill in the node maps for all beams
        idx = 0
        for beam in self.beams:
            for i in range(beam.num_nodes):
                beam.map.append(idx)
                idx += 6

        # re-assign joint nodes
        for joint in self.joints:
            members = joint['members']
            nodes = joint['nodes']

            # every joint node gets the index of the node in the first joint member
            for i, member in enumerate(members[1:]):
                member.map[nodes[i]] = members[0].map[nodes[0]]

        nodes = set()
        for beam in self.beams:
            for i in range(beam.num_nodes):
                nodes.add(beam.map[i])
        
        # the global dimension is the number of unique nodes times
        # the degrees of freedom per node
        dim = len(nodes) * 6

        
        # create the global stiffness matrix
        # and the global mass matrix
        K = np.zeros((dim, dim))
        M = np.zeros((dim, dim))

        for beam in self.beams:
            # lists of stiffness matrices for each element in the global frame
            transformed_stiffness_matrices = beam._transform_stiffness_matrices()
            transformed_mass_matrices = beam._transform_mass_matrices()
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


        # assemble te global loads vector
        F = np.zeros((dim))
        for beam in self.beams:
            loads = beam.loads
            map = beam.map

            for i in range(beam.num_nodes):
                idx = map[i]
                F[idx:idx+6] += loads[i, :]

        
        # add any inertial loads


        


        # apply boundary conditions
        for beam in self.beams:
            for node in beam.boundary_conditions:
                idx = beam.map[node]

                for i in range(6):
                    # if dof[i] == 1:
                        # zero the row/column then put a 1 in the diagonal
                        K[idx + i, :] = 0
                        K[:, idx + i] = 0
                        K[idx + i, idx + i] = 1

                        # zero the corresponding load index as well
                        F[idx + i] = 0



        # solve the system of equations
        U = np.linalg.solve(K, F)


        # find the displacements
        displacement = {}
        for beam in self.beams:
            displacement[beam.name] = np.zeros((beam.num_nodes, 3))

            for i in range(beam.num_nodes):
                idx = beam.map[i]
                # extract the (x, y, z) nodal displacement
                displacement[beam.name][i, :] = U[idx:idx+3]


        # calculate the elemental loads and stresses
        stress = {}
        for beam in self.beams:
            # elemental loads
            element_loads = beam._recover_loads(U)
            # perform a stress recovery
            beam_stress = np.zeros((beam.num_elements))
            for i in range(beam.num_elements):
                beam_stress[i] = beam.cs.stress(element_loads[i], i)

            stress[beam.name] = beam_stress







    