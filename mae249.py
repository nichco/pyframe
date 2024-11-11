import numpy as np
import pyframe as pf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 12
mesh = np.zeros((n, 3))
mesh[:, 1] = np.linspace(0, 10, n)

aluminum = pf.Material(E=70e9, G=26e9, density=2700)

# radius = np.ones(n - 1) * 0.5
radius = np.linspace(0.5, 0.3, n - 1)
# thickness = np.ones(n - 1) * 0.001
thickness = np.linspace(0.01, 0.001, n - 1)
cs = pf.CSTube(radius=radius, thickness=thickness)

loads = np.zeros((n, 6))
loads[:, 2] = np.linspace(1000, 500, n)

frame = pf.Frame()

beam_1 = pf.Beam(name='beam_1', mesh=mesh, material=aluminum, cs=cs)
beam_1.fix(0)
beam_1.add_load(loads)

frame.add_beam(beam_1)


# acc = np.array([0, 0, -9.81, 0, 0, 0])
# frame.add_acc(acc)



frame.solve()

displacement = frame.displacement['beam_1']
# print(displacement)

# test_stress = frame.beam_stress(beam_1, frame.U)
# print('test stress', test_stress)

nt = 300
start = 0
stop = 15
sim = pf.Simulation(frame, start=start, stop=stop, nt=nt)
t, u = sim.solve()
def_mesh = sim.parse_u(u, beam_1)

# sim.create_frames([def_mesh], xlim=[-1, 11], ylim=[-1.5, 1.5], figsize=(6, 3))
# sim.gif(filename='beam.gif', fps=10)

stress = np.zeros((nt, beam_1.num_elements))
for i in range(nt):
    beam_stress = frame.beam_stress(beam_1, u[:, i])
    stress[i, :] = beam_stress
    # print('beam stress', beam_stress)




time = np.linspace(start, stop, nt) 
# spanwise_index = np.linspace(0, beam_1.num_elements)
spanwise_index = np.linspace(-(beam_1.num_elements - 1), beam_1.num_elements - 1, 2 * beam_1.num_elements)
time_grid, spanwise_grid = np.meshgrid(time, spanwise_index)

reversed_vector = stress[:, ::-1]
stress = np.concatenate((reversed_vector, stress), axis=1)

# Create the figure
fig = plt.figure(figsize=(14, 6))

# Plot for extension strain
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(time_grid, spanwise_grid, stress.T * 1E-7 * 4500, cmap='jet')
ax1.set_xlabel('Time, s')
ax1.set_ylabel('Spanwise element index')
ax1.set_zlabel('Extension strain')

# ax1.set_xlim([start+0.5, stop])

plt.show()
