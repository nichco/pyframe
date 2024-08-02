import pyframe as pf
import numpy as np
import csdl_alpha as csdl


recorder = csdl.Recorder(inline=True)
recorder.start()

n = 21
mesh = np.zeros((n, 3))
mesh[:, 1] = np.linspace(0, 10, n)
mesh = csdl.Variable(value=mesh)

aluminum = pf.Material(E=70e9, G=26e9, density=2700)

radius = np.ones(n - 1) * 0.5
thickness = np.ones(n - 1) * 0.001
cs = pf.CSDLCSTube(radius=radius, thickness=thickness)

loads = np.zeros((n, 6))
loads[:, 2] = 20000
loads = csdl.Variable(value=loads)

beam = pf.CSDLBeam(name='beam_1', mesh=mesh, material=aluminum, cs=cs)
beam.fix(0)

frame = pf.CSDLFrame()
frame.add_beam(beam)

acc = csdl.Variable(value=np.array([0, 0, -9.81, 0, 0, 0]))
frame.add_acc(acc)

solution = frame.solve()
disp = solution.displacement['beam_1']

recorder.stop()

print(disp.value)
