import os

from accelerate_simulations.geometry import AbstractGeometry, make_msh


cl_coarse = 5.0
cl_fine = 2.0
n_timesteps = 50

b_1 = [100.0]
y_1 = [10.0]

b_2 = [100.0]
y_2 = [1.0]

circle_location_seed_list = range(13, 30)
n_circles_list = [5]
circle_radius_list = [5.0]
box_size_list = [(100, 100)]

path_data_dir = os.path.join('data', 'raw')

path_fem = os.path.join('fem', 'build', 'fem')
if not os.path.exists(path_fem):
    raise Exception('build fem first')


def make_directories_for(case_name):
    path_case_dir = os.path.join(path_data_dir, case_name)

    if os.path.exists(path_case_dir): 
        raise Exception(f'{path_case_dir} exists, remove it manually')

    paths = {
        'mesh': os.path.join(path_case_dir, 'mesh'),
        'mesh_hex': os.path.join(path_case_dir, 'mesh_hex.msh'), 
        'abstract_geometry': os.path.join(path_case_dir, 'abstract_geometry.pickle'), 
        'solution_dir': os.path.join(path_case_dir, 'solution/'), 
        'plastic_strain_dir': os.path.join(path_case_dir, 'plastic_strain/')
    }

    os.makedirs(path_case_dir)
    os.makedirs(paths['solution_dir'])
    os.makedirs(paths['plastic_strain_dir'])

    return paths


for i, circle_location_seed in enumerate(circle_location_seed_list):
    case_name = str(circle_location_seed)

    paths = make_directories_for(case_name)

    abstract_geometry = AbstractGeometry(
        n_circles_list[0],
        circle_radius_list[0],
        box_size_list[0],
        circle_location_seed)

    abstract_geometry.save_at(paths['abstract_geometry'])

    make_msh(
        paths['mesh'], abstract_geometry, 
        cl_coarse=cl_coarse, cl_fine=cl_fine, verbose=False)

    fem_args = [
        paths['mesh_hex'],
        paths['solution_dir'],
        paths['plastic_strain_dir'],
        str(b_1[0]),
        str(y_1[0]),
        str(b_2[0]),
        str(y_2[0]),
        str(n_timesteps)
    ]
    
    fem_args = ' '.join(fem_args)
    command = f'{path_fem} {fem_args}'
    os.system(command)
