import numpy as np
import pyframe as pf
import pyvista as pv
import pickle
import time
from scipy.optimize import minimize, LinearConstraint

with open('lunar_lander_meshes.pkl', 'rb') as file:
    meshes, radius = pickle.load(file)

x_scale = 100
c_scale = 20

n = meshes.shape[1]
ne = n - 1
aluminum = pf.Material(E=69E9, G=26E9, density=2700)


acc = np.array([0, 0, -9.81 * 40, 0, 0, 0])


def fun(x):
    frame = pf.Frame()
    beams = []

    for i in range(28):
        thickness = np.abs(x[i] / x_scale)
        beam_radius = np.ones(n - 1) * 0.03#radius[i]
        cs = pf.CSTube(radius=beam_radius, thickness=thickness)
        beam = pf.Beam(name='beam_'+str(i), mesh=meshes[i, :, :], material=aluminum, cs=cs)

        if i in [0, 4, 6, 10]: beam.fix(0)
        if i in [20, 21, 22, 23]: beam.add_inertial_mass(100, 0)
        if i in [24, 25, 26, 27]: beam.add_inertial_mass(50, 0)

        beams.append(beam)
        frame.add_beam(beam)


    # frame.add_joint(members=[beams[0], beams[1], beams[2]], nodes=[0, 0, 0])
    # frame.add_joint(members=[beams[3], beams[4], beams[5]], nodes=[0, 0, 0])
    # frame.add_joint(members=[beams[6], beams[7], beams[8]], nodes=[0, 0, 0])
    # frame.add_joint(members=[beams[9], beams[10], beams[11]], nodes=[0, 0, 0])
    # frame.add_joint(members=[beams[1], beams[3], beams[13], beams[14], beams[20], beams[21]], nodes=[ne, ne, 0, 0, ne, 0])
    # frame.add_joint(members=[beams[4], beams[6], beams[15], beams[17], beams[21], beams[22]], nodes=[ne, ne, 0, 0, ne, 0])
    # frame.add_joint(members=[beams[7], beams[9], beams[16], beams[18], beams[22], beams[23]], nodes=[ne, ne, 0, 0, ne, 0])
    # frame.add_joint(members=[beams[10], beams[0], beams[12], beams[19], beams[20], beams[23]], nodes=[ne, ne, 0, 0, 0, ne])
    # frame.add_joint(members=[beams[2], beams[12], beams[13], beams[24], beams[25]], nodes=[ne, ne, ne, ne, 0])
    # frame.add_joint(members=[beams[5], beams[14], beams[15], beams[25], beams[26]], nodes=[ne, ne, ne, ne, 0])
    # frame.add_joint(members=[beams[8], beams[16], beams[17], beams[26], beams[27]], nodes=[ne, ne, ne, ne, 0])
    # frame.add_joint(members=[beams[11], beams[18], beams[19], beams[27], beams[24]], nodes=[ne, ne, ne, ne, 0])

    # frame.add_acc(acc)

    # solution = frame.solve()

    mass = frame.compute_mass()

    return mass * 1



def con(x):

    frame = pf.Frame()
    beams = []

    for i in range(28):
        thickness = np.abs(x[i] / x_scale)
        beam_radius = np.ones(n - 1) * radius[i]
        cs = pf.CSTube(radius=beam_radius, thickness=thickness)
        beam = pf.Beam(name='beam_'+str(i), mesh=meshes[i, :, :], material=aluminum, cs=cs)

        if i in [0, 4, 6, 10]: beam.fix(0)
        if i in [20, 21, 22, 23]: beam.add_inertial_mass(100, 0)
        if i in [24, 25, 26, 27]: beam.add_inertial_mass(50, 0)

        beams.append(beam)
        frame.add_beam(beam)


    frame.add_joint(members=[beams[0], beams[1], beams[2]], nodes=[0, 0, 0])
    frame.add_joint(members=[beams[3], beams[4], beams[5]], nodes=[0, 0, 0])
    frame.add_joint(members=[beams[6], beams[7], beams[8]], nodes=[0, 0, 0])
    frame.add_joint(members=[beams[9], beams[10], beams[11]], nodes=[0, 0, 0])
    frame.add_joint(members=[beams[1], beams[3], beams[13], beams[14], beams[20], beams[21]], nodes=[ne, ne, 0, 0, ne, 0])
    frame.add_joint(members=[beams[4], beams[6], beams[15], beams[17], beams[21], beams[22]], nodes=[ne, ne, 0, 0, ne, 0])
    frame.add_joint(members=[beams[7], beams[9], beams[16], beams[18], beams[22], beams[23]], nodes=[ne, ne, 0, 0, ne, 0])
    frame.add_joint(members=[beams[10], beams[0], beams[12], beams[19], beams[20], beams[23]], nodes=[ne, ne, 0, 0, 0, ne])
    frame.add_joint(members=[beams[2], beams[12], beams[13], beams[24], beams[25]], nodes=[ne, ne, ne, ne, 0])
    frame.add_joint(members=[beams[5], beams[14], beams[15], beams[25], beams[26]], nodes=[ne, ne, ne, ne, 0])
    frame.add_joint(members=[beams[8], beams[16], beams[17], beams[26], beams[27]], nodes=[ne, ne, ne, ne, 0])
    frame.add_joint(members=[beams[11], beams[18], beams[19], beams[27], beams[24]], nodes=[ne, ne, ne, ne, 0])

    frame.add_acc(acc)

    solution = frame.solve()

    stress = np.zeros((28, n - 1))
    for i in range(28):
        stress[i, :] = solution.stress['beam_'+str(i)]

    stress = stress.flatten() * 1E-6
    print(np.max(stress))

    limit = 60
    return (limit - stress) / c_scale



# x0 = np.ones(28) * 0.001
x0 = np.ones(28) * 0.01 * x_scale
bnds = ((0.001, 0.015),) * 28

res = minimize(fun, 
               x0, 
               method='SLSQP', 
               options={'ftol': 1e-4, 'disp': True, 'maxiter': 50}, 
               bounds=bnds, 
               constraints={'type': 'ineq', 'fun': con})
print(res.x)