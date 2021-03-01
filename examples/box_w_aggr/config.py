import os

# --- fem model parameters
n_timesteps = 50
circle_location_seed_list = range(18)

# --- nn parameters
resolution = (128, 128)

field_names = ['plastic_strain'] # ['plastic_strain', 'solution']
time_step = 50

batch_size = 4

# --- geometrical parameters
box_size = (200, 200)
circle_radius_range = (20, 50)
gap = 10
circle_density = 0.90

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

