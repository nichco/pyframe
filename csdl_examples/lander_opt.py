import numpy as np
import pyframe as pf
# import pyvista as pv
import pickle
import time
# from scipy.optimize import minimize
from scipy.optimize import differential_evolution


with open('lunar_lander_meshes.pkl', 'rb') as file:
    meshes, radius = pickle.load(file)


c_scale = 20

n = meshes.shape[1]
ne = n - 1
aluminum = pf.Material(E=69E9, G=26E9, density=2700)


acc = np.array([0, 0, -9.81 * 40, 0, 0, 0])
# [0.00053907 0.00014542 0.00040952]
# [0.00054388 0.00014725 0.00040671]

def fun(x):
    frame = pf.Frame()
    beams = []

    for i in range(28):

        if i in[2, 5, 8, 11]:
            thickness = x[0]
        elif i in [12, 13, 14, 15, 16, 17, 18, 19]:
            thickness = x[1]
        else:
            thickness = x[2]



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

    mass = frame.compute_mass()
    obj = mass

    limit = 0.05

    disp = np.zeros((28, n))
    for i in range(28):
        u = np.linalg.norm(solution.displacement['beam_'+str(i)], axis=1)
        for j in range(n):
            if u[j] > limit:
                obj += 300
        disp[i, :] = u
        # if u > limit:
        #     obj += 1000

    return obj



# x0 = np.ones(28) * 0.001
bnds = ((1E-5, 0.015),) * 3#28

# res = minimize(fun, 
#                x0, 
#                method='SLSQP', 
#                options={'ftol': 1e-4, 'disp': True, 'maxiter': 50}, 
#                bounds=bnds, 
#                constraints={'type': 'ineq', 'fun': con})
# print(res.x)

if __name__ == '__main__':
    t1 = time.time()
    res = differential_evolution(fun, 
                                 bounds=bnds, 
                                 disp=True, 
                                 workers=-1, 
                                 popsize=4, # lower is faster, but less likely to find the global minimum
                                 recombination=0.5, # higher increases exploration. Lower refines the current best solution
                                 tol=1E-4)
    # res = shgo(fun, bounds=bnds, options={'disp': True}, workers=1)
    # res = minimize(fun, x0, method='nelder-mead', bounds=bnds, options={'xatol': 1e-8, 'disp': True})
    t2 = time.time()
    print('time: ', t2 - t1)
    print(res.x)