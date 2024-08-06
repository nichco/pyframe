import csdl_alpha as csdl
import numpy as np
import pyframe as pf
import pyvista as pv
from modopt import CSDLAlphaProblem
from modopt import SLSQP
import pickle

with open('lunar_lander_meshes.pkl', 'rb') as file:
    meshes, _ = pickle.load(file)

n = meshes.shape[1]


aluminum = pf.Material(E=69E9, G=26E9, density=2700)


recorder = csdl.Recorder(inline=True, debug=False)
recorder.start()

# aluminum = pf.Material(E=69E9, G=26E9, density=2700)

frame = pf.CSDLFrame()

beam_meshes = []

for i in range(24):
    beam_mesh = meshes[i, :, :]
    beam_mesh = csdl.Variable(value=beam_mesh)
    beam_meshes.append(beam_mesh)

beams = []

r = csdl.Variable(value=np.ones((24)) * 0.1)
print(r.value)
r.set_as_design_variable(lower=0.01, upper=0.5, scaler=1E1)
radii = csdl.expand(r, (24, n-1), 'i->ij')

for i in range(24):
    # r = csdl.Variable(value=0.1)
    # r.set_as_design_variable(lower=0.01, scaler=1E1)
    # radius = csdl.expand(r[i], (n - 1,))
    beam_cs = pf.CSDLCSCircle(radius=radii[i, :])
    beam = pf.CSDLBeam(name='beam_'+str(i), mesh=beam_meshes[i], material=aluminum, cs=beam_cs)

    if i in [0, 4, 6, 10]: # fix the feet
        beam.fix(0)

    if i in [20, 21, 22, 23]: # add mass to bottom frame
        beam.add_inertial_mass(100, 0)

    beams.append(beam)
    frame.add_beam(beam)



ne = n - 1
# foot joints
frame.add_joint(members=[beams[0], beams[1], beams[2]], nodes=[0, 0, 0])
frame.add_joint(members=[beams[3], beams[4], beams[5]], nodes=[0, 0, 0])
frame.add_joint(members=[beams[6], beams[7], beams[8]], nodes=[0, 0, 0])
frame.add_joint(members=[beams[9], beams[10], beams[11]], nodes=[0, 0, 0])
# middle outside joints
frame.add_joint(members=[beams[1], beams[3], beams[13], beams[14]], nodes=[ne, ne, 0, 0, ne, 0])
frame.add_joint(members=[beams[4], beams[6], beams[15], beams[17]], nodes=[ne, ne, 0, 0, ne, 0])
frame.add_joint(members=[beams[7], beams[9], beams[16], beams[18]], nodes=[ne, ne, 0, 0, ne, 0])
frame.add_joint(members=[beams[10], beams[0], beams[12], beams[19]], nodes=[ne, ne, 0, 0, 0, ne])
# leg strut attach joints
frame.add_joint(members=[beams[2], beams[12], beams[13], beams[20], beams[21]], nodes=[ne, ne, ne, ne, 0])
frame.add_joint(members=[beams[5], beams[14], beams[15], beams[21], beams[22]], nodes=[ne, ne, ne, ne, 0])
frame.add_joint(members=[beams[8], beams[16], beams[17], beams[22], beams[23]], nodes=[ne, ne, ne, ne, 0])
frame.add_joint(members=[beams[11], beams[18], beams[19], beams[23], beams[20]], nodes=[ne, ne, ne, ne, 0])


acc = csdl.Variable(value=np.array([0, 0, -9.81 * 30, 0, 0, 0]))
frame.add_acc(acc)

solution = frame.solve()



disp = csdl.Variable(value=np.zeros((28, 4, 3)))
for i, beam in enumerate(beams):
    beam_disp = solution.displacement[beam.name]
    disp = disp.set(csdl.slice[i, :, :], beam_disp)

ndisp = csdl.norm(disp + 1E-6, axes=(2,))
# squared_disp = disp**2

max_disp = csdl.maximum(ndisp * 10)/10
max_disp.set_as_constraint(upper=0.03, scaler=1E1)

# min_disp = csdl.maximum(-disp)
# min_disp.set_as_constraint(upper=0.1, scaler=1E1)


mass = frame.compute_mass()
mass.set_as_objective(scaler=1E-2)

recorder.stop()

# recorder.count_operations()

# import tracemalloc
# tracemalloc.start()

import time
t1 = time.time()

sim = csdl.experimental.PySimulator(recorder)
# sim = csdl.experimental.JaxSimulator(recorder=recorder) # 22850 & 98330
# sim.run()
prob = CSDLAlphaProblem(problem_name='lander', simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 100, 'ftol': 1e-5, 'disp': True})
optimizer.solve()
optimizer.print_results()

t2 = time.time()
print('time: ', t2 - t1)

# time: 140.84449791908264 with einsums in transform
# time: 113 with even more einsums
# time: 112 with condition extra inertial loads
# time: 107 by moving some ops out of transformations for loop
# time: 87 by moving even more out of transofrmations for loop
# time: 75 with .set() called outside of for loops
# time: 73 small changes

# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()

print('mass: ', mass.value)
print('r: ', r.value)


# exit()
plotter = pv.Plotter()

for i, beam in enumerate(frame.beams):
    mesh0 = beam.mesh.value
    disp = solution.displacement[beam.name].value
    mesh1 = mesh0 + 20 * disp

    radius = beam.cs.radius.value
    # radius = np.ones((n - 1))*0.1

    disp = np.linalg.norm(solution.displacement[beam.name].value, axis=1)

    # af.plot_mesh(plotter, mesh0, color='lightblue', line_width=10)
    # plot_mesh(plotter, mesh1, cell_data=stress, cmap='viridis', line_width=20)
    pf.plot_points(plotter, mesh1, color='blue', point_size=15)
    pf.plot_cyl(plotter, mesh1, cell_data=None, radius=radius, cmap='plasma')

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