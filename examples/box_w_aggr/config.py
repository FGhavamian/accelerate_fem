import os

# --- fem model parameters
n_timesteps = 50

# --- geometrical parameters
circle_radius_range = (3, 7)
n_circles = 10
box_size = (100, 100)
gap = 10

cl_coarse = 5.0
cl_fine = 2.0

circle_location_seed_list = range(1, 30)

# --- material parameters
b_1 = 100.0
y_1 = 10.0
b_2 = 100.0
y_2 = 1.0


# --- paths
path_data_dir = os.path.join('data', 'raw')

path_fem = os.path.join('..', '..', 'fem', 'build', 'fem')
if not os.path.exists(path_fem):
    raise Exception('build fem first')

