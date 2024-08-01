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

frame.add_beam(beam_1)


acc = np.array([0, 0, -9.81, 0, 0, 0])
frame.add_acc(acc)



solution = frame.solve()

displacement = solution.displacement['beam_1']
print(displacement)