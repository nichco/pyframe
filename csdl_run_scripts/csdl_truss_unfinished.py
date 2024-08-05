import pyframe as pf
import numpy as np
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
import cProfile
from pstats import SortKey


recorder = csdl.Recorder(inline=True)
recorder.start()

n = 4

# top right upright
mesh = np.zeros((n, 3))
mesh[:, 0] = 0.5
mesh[:, 1] = 0.5
mesh[:, 2] = np.linspace(0, 1, n)
mesh_1 = csdl.Variable(value=mesh)

# top left upright
mesh = np.zeros((n, 3))
mesh[:, 0] = -0.5
mesh[:, 1] = 0.5
mesh[:, 2] = np.linspace(0, 1, n)
mesh_2 = csdl.Variable(value=mesh)

# bottom left upright
mesh = np.zeros((n, 3))
mesh[:, 0] = -0.5
mesh[:, 1] = -0.5
mesh[:, 2] = np.linspace(0, 1, n)
mesh_2 = csdl.Variable(value=mesh)

# bottom right upright
mesh = np.zeros((n, 3))
mesh[:, 0] = -0.5
mesh[:, 1] = -0.5
mesh[:, 2] = np.linspace(0, 1, n)
mesh_2 = csdl.Variable(value=mesh)

# right upright
mesh = np.zeros((n, 3))
mesh[:, 0] = 5
mesh[:, 1] = np.linspace(0, 10, n)
mesh_2 = csdl.Variable(value=mesh)

# top
mesh = np.zeros((n, 3))
mesh[:, 0] = np.linspace(0, 5, n)
mesh[:, 1] = 10
mesh_3 = csdl.Variable(value=mesh)

aluminum = pf.Material(E=70e9, G=26e9, density=2700)

r_1 = csdl.Variable(value=0.1)
r_1.set_as_design_variable(lower=0.001, scaler=1E1)
cs_1 = pf.CSDLCSCircle(radius=csdl.expand(r_1, (n-1,)))

r_2 = csdl.Variable(value=0.1)
r_2.set_as_design_variable(lower=0.001, scaler=1E1)
cs_2 = pf.CSDLCSCircle(radius=csdl.expand(r_2, (n-1,)))

r_3 = csdl.Variable(value=0.1)
r_3.set_as_design_variable(lower=0.001, scaler=1E1)
cs_3 = pf.CSDLCSCircle(radius=csdl.expand(r_3, (n-1,)))


beam_1 = pf.CSDLBeam(name='beam_1', mesh=mesh_1, material=aluminum, cs=cs_1)
beam_1.fix(0)

beam_2 = pf.CSDLBeam(name='beam_2', mesh=mesh_2, material=aluminum, cs=cs_2)
beam_2.fix(0)

beam_3 = pf.CSDLBeam(name='beam_3', mesh=mesh_3, material=aluminum, cs=cs_3)


frame = pf.CSDLFrame()
frame.add_beam(beam_1)
frame.add_beam(beam_2)
frame.add_beam(beam_3)

frame.add_joint(members=[beam_1, beam_3], nodes=[n-1, 0])
frame.add_joint(members=[beam_3, beam_2], nodes=[n-1, n-1])

acc = csdl.Variable(value=np.array([-100, 0, 0, 0, 0, 0]))
frame.add_acc(acc)

solution = frame.solve()

# upper_disp = 0.5
disp_1 = solution.displacement['beam_1']
max_disp_1 = csdl.norm(disp_1[-1, :])
max_disp_1.set_as_constraint(upper=0.5, scaler=1E1)

# disp_2 = solution.displacement['beam_2']
# disp_2.set_as_constraint(upper=0.5, scaler=1E1)

# disp_3 = solution.displacement['beam_3']

mass = frame.compute_mass()
mass.set_as_objective(scaler=1E-3)

recorder.stop()




# import time



# def fun():
#     t1 = time.time()

#     # sim = csdl.experimental.PySimulator(recorder)
#     sim = csdl.experimental.JaxSimulator(recorder=recorder)
#     # sim.run()
#     prob = CSDLAlphaProblem(problem_name='lander', simulator=sim)
#     optimizer = SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-5, 'disp': True})
#     print('solving')
#     optimizer.solve()
#     optimizer.print_results()

#     t2 = time.time()
#     print('time: ', t2 - t1)

# cProfile.run('fun()', 'profile_data')

# import pstats

# p = pstats.Stats('profile_data')
# # p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
# p.sort_stats(SortKey.TIME).print_stats(20)




# sim = csdl.experimental.PySimulator(recorder)
sim = csdl.experimental.JaxSimulator(recorder=recorder)
# sim.run()
prob = CSDLAlphaProblem(problem_name='lander', simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-6, 'disp': True})
optimizer.solve()
optimizer.print_results()





print('mass: ', mass.value)
print('r_1: ', r_1.value)
print('r_2: ', r_2.value)
print('r_3: ', r_3.value)
print('disp_1: ', disp_1.value)



def_mesh_1 = mesh_1.value + solution.displacement['beam_1'].value
def_mesh_2 = mesh_2.value + solution.displacement['beam_2'].value
def_mesh_3 = mesh_3.value + solution.displacement['beam_3'].value

plt.plot(def_mesh_1[:, 0], def_mesh_1[:, 1], 'black')
plt.scatter(def_mesh_1[:, 0], def_mesh_1[:, 1], color='black')
plt.plot(def_mesh_2[:, 0], def_mesh_2[:, 1], 'black')
plt.scatter(def_mesh_2[:, 0], def_mesh_2[:, 1], color='black')
plt.plot(def_mesh_3[:, 0], def_mesh_3[:, 1], 'black')
plt.scatter(def_mesh_3[:, 0], def_mesh_3[:, 1], color='black')
plt.axis('equal')
plt.grid()
plt.show()