import numpy as np


def distance_calc(mesh_in, mesh_out):
    mesh_in_shape = mesh_in.shape
    mesh_out_shape = mesh_out.shape

    # Compute distances between in_mesh and out_mesh
    # first we expand both meshes to fit the same shape
    in_mesh_exp = np.expand_dims(mesh_in, axis=1)
    in_mesh_exp = np.tile(in_mesh_exp, (1, mesh_out_shape[0], 1))
    out_mesh_exp = np.expand_dims(mesh_out, axis=0)
    out_mesh_exp = np.tile(out_mesh_exp, (mesh_in_shape[0], 1, 1))
    # then we subtract their coordinates
    mesh_dist_exp = in_mesh_exp - out_mesh_exp
    # lastly we compute the 2-norm along the last axis of mesh_dist_exp
    dist = np.linalg.norm(mesh_dist_exp, axis=2)

    return dist









if __name__ == '__main__':
    import numpy as np
    np.random.seed(1)

    in_shape_test = (5, 3)
    out_shape_test = (8, 3)

    # define random solid and fluid mesh coordinates for testing
    rng_solid_mesh = np.random.random(in_shape_test)
    rng_fluid_mesh = np.random.random(out_shape_test)

    # compare with numpy implementation
    distance_array_np = distance_calc(rng_solid_mesh, rng_fluid_mesh)
    print("Distance array numpy:")
    print(distance_array_np)