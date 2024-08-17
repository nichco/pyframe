import pyframe as pf
import numpy as np


n = 21
mesh = np.zeros((n, 3))
mesh[:, 1] = np.linspace(0, 10, n)

aluminum = pf.Material(E=70e9, G=26e9, density=2700)

radius = np.ones(n - 1) * 0.5
thickness = np.ones(n - 1) * 0.001
cs = pf.CSTube(radius=radius, thickness=thickness)

loads = np.zeros((n, 6))
loads[:, 2] = 20000

frame = pf.Frame()

beam_1 = pf.Beam(name='beam_1', mesh=mesh, material=aluminum, cs=cs)
beam_1.fix(0)
beam_1.add_load(loads)

beam_2 = pf.Beam(name='beam_2', mesh=mesh, material=aluminum, cs=cs)
# beam_2.fix(0)
beam_2.add_load(loads)

frame.add_beam(beam_1)
frame.add_beam(beam_2)


acc = np.array([0, 0, -9.81, 0, 0, 0])
frame.add_acc(acc)

frame.add_joint(members=[beam_1, beam_2], nodes=[n-1, n-1])



solution = frame.solve()

displacement1 = solution.displacement['beam_1']
displacement2 = solution.displacement['beam_2']
print(displacement1)


natural_frequency = frame.compute_natural_frequency()
print(natural_frequency)

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)


def_mesh1 = mesh + displacement1
plt.plot(def_mesh1[:, 1], def_mesh1[:, 2])
def_mesh2 = mesh + displacement2
plt.plot(def_mesh2[:, 1], def_mesh2[:, 2])
plt.show()