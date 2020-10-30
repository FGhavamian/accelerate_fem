import os

cl_coarse = 5.0
cl_fine = 2.0
n_timesteps = 50

b_1 = [100.0]
y_1 = [10.0]

b_2 = [100.0]
y_2 = [1.0]

circle_location_seed_list = range(0, 25)
n_circles_list = [5]
circle_radius_list = [5.0]
box_size_list = [(100, 100)]

path_data_dir = os.path.join('data', 'raw')

path_to_tethex = "~/dev/tethex/build/tethex"

path_fem = os.path.join('..', '..', 'fem', 'build', 'fem')
if not os.path.exists(path_fem):
    raise Exception('build fem first')

