import numpy as np
import pyframe as pf
import pyvista as pv
import pickle
import cProfile
from pstats import SortKey
import tracemalloc
tracemalloc.start()

with open('lunar_lander_meshes.pkl', 'rb') as file:
    meshes, radius = pickle.load(file)

n = meshes.shape[1]

aluminum = pf.Material(E=69E9, G=26E9, density=2700)

frame = pf.Frame()

beams = []

for i in range(28):
    thickness = 0.001
    beam_radius = np.ones(n - 1) * radius[i]
    cs = pf.CSTube(radius=beam_radius, thickness=thickness)
    beam = pf.Beam(name='beam_'+str(i), mesh=meshes[i, :, :], material=aluminum, cs=cs)

    if i in [0, 4, 6, 10]: # fix the feet
        beam.fix(0)

    if i in [20, 21, 22, 23]: # add mass to bottom frame
        beam.add_inertial_mass(100, 0)

    if i in [24, 25, 26, 27]:  # add mass to top frame
        beam.add_inertial_mass(50, 0)

    beams.append(beam)
    frame.add_beam(beam)




ne = n - 1
# foot joints
frame.add_joint(members=[beams[0], beams[1], beams[2]], nodes=[0, 0, 0])
frame.add_joint(members=[beams[3], beams[4], beams[5]], nodes=[0, 0, 0])
frame.add_joint(members=[beams[6], beams[7], beams[8]], nodes=[0, 0, 0])
frame.add_joint(members=[beams[9], beams[10], beams[11]], nodes=[0, 0, 0])
# middle outside joints
frame.add_joint(members=[beams[1], beams[3], beams[13], beams[14], beams[20], beams[21]], nodes=[ne, ne, 0, 0, ne, 0])
frame.add_joint(members=[beams[4], beams[6], beams[15], beams[17], beams[21], beams[22]], nodes=[ne, ne, 0, 0, ne, 0])
frame.add_joint(members=[beams[7], beams[9], beams[16], beams[18], beams[22], beams[23]], nodes=[ne, ne, 0, 0, ne, 0])
frame.add_joint(members=[beams[10], beams[0], beams[12], beams[19], beams[20], beams[23]], nodes=[ne, ne, 0, 0, 0, ne])
# leg strut attach joints
frame.add_joint(members=[beams[2], beams[12], beams[13], beams[24], beams[25]], nodes=[ne, ne, ne, ne, 0])
frame.add_joint(members=[beams[5], beams[14], beams[15], beams[25], beams[26]], nodes=[ne, ne, ne, ne, 0])
frame.add_joint(members=[beams[8], beams[16], beams[17], beams[26], beams[27]], nodes=[ne, ne, ne, ne, 0])
frame.add_joint(members=[beams[11], beams[18], beams[19], beams[27], beams[24]], nodes=[ne, ne, ne, ne, 0])



acc = np.array([0, 0, -9.81 * 40, 0, 0, 0])
frame.add_acc(acc)

import time
# t1 = time.time()
# solution = frame.solve()
# t2 = time.time()
# print(t2 - t1)

# exit()
def solve():
    t1 = time.time()
    solution = frame.solve()
    t2 = time.time()
    print('time: ', t2 - t1)
    return solution

cProfile.run('solve()', 'profile_data')

import pstats

p = pstats.Stats('profile_data')
# p.sort_stats(SortKey.CUMULATIVE).print_stats(20)
p.sort_stats(SortKey.TIME).print_stats(25)

# 4986 function calls (4930 primitive calls) in 0.096 seconds
# 0.08944296836853027



print(tracemalloc.get_traced_memory())
tracemalloc.stop()


exit()

plotter = pv.Plotter()

for i, beam in enumerate(frame.beams):
    mesh0 = beam.mesh
    disp = solution.displacement[beam.name]
    mesh1 = mesh0 + 20 * disp

    radius = beam.cs.radius

    stress = solution.stress[beam.name]

    # af.plot_mesh(plotter, mesh0, color='lightblue', line_width=10)
    # plot_mesh(plotter, mesh1, cell_data=stress, cmap='viridis', line_width=20)
    pf.plot_points(plotter, mesh1, color='blue', point_size=15)

    # radius = np.ones((beam.num_elements)) * 0.1
    pf.plot_cyl(plotter, mesh1, cell_data=stress, radius=radius, cmap='plasma')

    if i in [0, 4, 6, 10]:
        cyl = pv.Cylinder(center=mesh1[0, :], direction=[0, 0, 1], radius=0.6, height=0.2)
        plotter.add_mesh(cyl, color='thistle')



zo = np.array([0, 0, -0.2])
scale = 70

ft1 = pv.read('stl/foot.stl')
ft1.scale(scale, inplace=True)
ft1.translate(meshes[0,0,:] + zo, inplace=True)
plotter.add_mesh(ft1, color='red')

ft2 = pv.read('stl/foot.stl')
ft2.scale(scale, inplace=True)
ft2.translate(meshes[4,0,:] + zo, inplace=True)
plotter.add_mesh(ft2, color='red')

ft3 = pv.read('stl/foot.stl')
ft3.scale(scale, inplace=True)
ft3.translate(meshes[6,0,:] + zo, inplace=True)
plotter.add_mesh(ft3, color='red')

ft4 = pv.read('stl/foot.stl')
ft4.scale(scale, inplace=True)
ft4.translate(meshes[10,0,:] + zo, inplace=True)
plotter.add_mesh(ft4, color='red')

# toroidal tank
torus = pv.ParametricTorus(ringradius=2.375, crosssectionradius=1.4, center=(0, 0, 3.5))
plotter.add_mesh(torus, color='skyblue', opacity=0.5)

# engine
eng = pv.read('stl/j2engine.stl')
eng.scale(0.01, inplace=True)
eng.translate([0, 0, 1], inplace=True)
plotter.add_mesh(eng, color='orange')

# top tank
sph = pv.Sphere(center=(0, 0, 5.25), radius=1.5)
plotter.add_mesh(sph, color='beige', opacity=0.5)

plotter.show()