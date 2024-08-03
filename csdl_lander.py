import csdl_alpha as csdl
import numpy as np
import pyframe as pf
import pyvista as pv
from modopt import CSDLAlphaProblem
from modopt import SLSQP
import pickle

with open('lunar_lander_meshes.pkl', 'rb') as file:
    meshes, radius = pickle.load(file)

n = meshes.shape[1]



recorder = csdl.Recorder(inline=False, debug=False)
recorder.start()

aluminum = pf.Material(E=69E9, G=26E9, density=2700)

frame = pf.CSDLFrame()

beam_meshes = []

for i in range(28):
    beam_mesh = meshes[i, :, :]
    beam_mesh = csdl.Variable(value=beam_mesh)
    beam_meshes.append(beam_mesh)

beam_radius = []

for i in range(28):
    # n = beam_meshes[i].shape[0]
    r = csdl.Variable(value=np.ones(n - 1) * radius[i])
    beam_radius.append(r)

beams = []
beam_thicknesses = []

for i in range(28):
    t = csdl.Variable(value=0.005)
    beam_thicknesses.append(t)
    t.set_as_design_variable(lower=0.001, upper=radius[i], scaler=1E2)
    beam_thickness = csdl.expand(t, (n - 1,))
    beam_cs = pf.CSDLCSTube(radius=beam_radius[i], thickness=beam_thickness)
    beam = pf.CSDLBeam(name='beam_'+str(i), mesh=beam_meshes[i], material=aluminum, cs=beam_cs)

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


acc = csdl.Variable(value=np.array([0, 0, -9.81 * 30, 0, 0, 0]))
frame.add_acc(acc)

solution = frame.solve()


# frame_stress = csdl.Variable(value=np.zeros((28, ne)))
# for i, beam in enumerate(beams):
#     stress = solution.get_stress(beam)
#     frame_stress = frame_stress.set(csdl.slice[i, :], stress)

# max_stress = csdl.maximum(frame_stress * 1E-6) # need to flatten it?
# max_stress.set_as_constraint(upper=20, scaler=1E-7)

limit = 200

stress = csdl.Variable(value=np.zeros((28, n - 1)))
for i in range(28):
    stress = stress.set(csdl.slice[i, :], solution.stress['beam_'+str(i)])

max_stress = csdl.maximum(stress * 1E-6)

# phi = (stress * 1E-6) - limit
# penalty = csdl.inner(phi, phi)

max_stress.set_as_constraint(upper=50, scaler=1E-7)

mass = solution.mass #+ penalty
mass.set_as_objective(scaler=1E-2)

recorder.stop()

# recorder.count_operations()
# SetVarIndex : 3392
# Reshape : 3002
# GetVarIndex : 2948
# BroadcastMult : 2493
# Add : 1125
# Mult : 1064
# Mult : 1064
# RightBroadcastPower : 1064
# BroadcastSetIndex : 768
# Sub : 728
# Div : 672
# exit()
# import tracemalloc
# tracemalloc.start()

import time
t1 = time.time()

# sim = csdl.experimental.PySimulator(recorder)
sim = csdl.experimental.JaxSimulator(recorder=recorder)
sim.run()
# prob = CSDLAlphaProblem(problem_name='lander', simulator=sim)
# optimizer = SLSQP(prob, solver_options={'maxiter': 50, 'ftol': 1e-5, 'disp': True})
# print('solving')
# optimizer.solve()
# optimizer.print_results()

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
print(beam_thicknesses[0].value)


exit()
plotter = pv.Plotter()

for i, beam in enumerate(frame.beams):
    mesh0 = beam.mesh.value
    disp = solution.displacement[beam.name].value
    mesh1 = mesh0 + 20 * disp

    radius = beam.cs.radius.value

    stress = solution.stress[beam.name].value

    # af.plot_mesh(plotter, mesh0, color='lightblue', line_width=10)
    # plot_mesh(plotter, mesh1, cell_data=stress, cmap='viridis', line_width=20)
    pf.plot_points(plotter, mesh1, color='blue', point_size=15)
    pf.plot_cyl(plotter, mesh1, cell_data=stress, radius=radius, cmap='plasma')

    if i in [0, 4, 6, 10]:
        cyl = pv.Cylinder(center=mesh1[0, :], direction=[0, 0, 1], radius=0.6, height=0.2)
        plotter.add_mesh(cyl, color='thistle')



zo = np.array([0, 0, -0.2])
scale = 70

ft1 = pv.read('foot.stl')
ft1.scale(scale, inplace=True)
ft1.translate(meshes[0,0,:] + zo, inplace=True)
plotter.add_mesh(ft1, color='red')

ft2 = pv.read('foot.stl')
ft2.scale(scale, inplace=True)
ft2.translate(meshes[4,0,:] + zo, inplace=True)
plotter.add_mesh(ft2, color='red')

ft3 = pv.read('foot.stl')
ft3.scale(scale, inplace=True)
ft3.translate(meshes[6,0,:] + zo, inplace=True)
plotter.add_mesh(ft3, color='red')

ft4 = pv.read('foot.stl')
ft4.scale(scale, inplace=True)
ft4.translate(meshes[10,0,:] + zo, inplace=True)
plotter.add_mesh(ft4, color='red')

# toroidal tank
torus = pv.ParametricTorus(ringradius=2.375, crosssectionradius=1.4, center=(0, 0, 3.5))
plotter.add_mesh(torus, color='skyblue', opacity=0.5)

# engine
eng = pv.read('j2engine.stl')
eng.scale(0.01, inplace=True)
eng.translate([0, 0, 1], inplace=True)
plotter.add_mesh(eng, color='orange')

# top tank
sph = pv.Sphere(center=(0, 0, 5.25), radius=1.5)
plotter.add_mesh(sph, color='beige', opacity=0.5)

plotter.show()