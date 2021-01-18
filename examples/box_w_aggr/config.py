import os

# --- fem model parameters
n_timesteps = 50

cl_coarse = 5.0
cl_fine = 2.0
circle_location_seed_list = range(1, 30)

# --- nn parameters
resolution = (400, 400)

field_names = ['plastic_strain'] # ['plastic_strain', 'solution']
time_step = 40

batch_size = 4

# --- geometrical parameters
circle_radius_range = (3, 7)
n_circles = 10
box_size = (100, 100)
gap = 10

# --- material parameters
b_1 = 100.0
y_1 = 10.0
b_2 = 100.0
y_2 = 1.0

# --- parameters from input fields
names_boundary = ['circles_boundaries']

material_properties = {
    "y": {
        "circles": y_1,
        "circles_boundaries": y_1,
        "box_wo_circles": y_2
    },
    "b": {
        "circles": b_1,
        "circles_boundaries": b_1,
        "box_wo_circles": b_2
    }
}


# --- paths
path_data_raw = os.path.join('data', 'raw')
path_data_processed = os.path.join('data', 'processed')

path_fem = os.path.join('..', '..', 'fem', 'build', 'fem')
if not os.path.exists(path_fem):
    raise Exception('build fem first')

