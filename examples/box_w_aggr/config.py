import os


# data collection 
# --- varying geometrical parameters
# circle_location_seeds = range(10)
# circle_radius_range = [(5*(1+i), 5*(2+i)) for i in range(10)]

circle_location_seeds = [3]
circle_radius_range = [[15, 20]]

# --- static geometrical parameters
resolution = (128, 128)
box_size = (200, 200)
gap = 10
circle_density = 0.90

# --- static material parameters
b_1 = 100.0
y_1 = 10.0
b_2 = 100.0
y_2 = 1.0

# --- meta data
n_timesteps = 50
field_names = ['plastic_strain'] # ['plastic_strain', 'solution']

time_step = 50

# --- nn parameters
batch_size = 4


# preprocessing parameters
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

