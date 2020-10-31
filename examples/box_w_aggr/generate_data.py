import os

from accelerate_simulations.geometry import AbstractGeometry, make_msh
import config


def make_directories_for(case_name):
    path_case_dir = os.path.join(config.path_data_dir, case_name)

    if os.path.exists(path_case_dir): 
        raise Exception(f'{path_case_dir} exists, remove it manually')

    paths = {
        'mesh': os.path.join(path_case_dir, 'mesh.msh'),
        'abstract_geometry': os.path.join(path_case_dir, 'abstract_geometry.pickle'), 
        'solution_dir': os.path.join(path_case_dir, 'solution/'), 
        'plastic_strain_dir': os.path.join(path_case_dir, 'plastic_strain/')
    }

    os.makedirs(path_case_dir)
    os.makedirs(paths['solution_dir'])
    os.makedirs(paths['plastic_strain_dir'])

    return paths


for i, circle_location_seed in enumerate(config.circle_location_seed_list):
    case_name = str(circle_location_seed)

    paths = make_directories_for(case_name)

    abstract_geometry = AbstractGeometry(
        config.n_circles,
        config.circle_radius_range,
        config.box_size,
        config.gap,
        circle_location_seed)

    abstract_geometry.save_at(paths['abstract_geometry'])

    make_msh(
        paths['mesh'], 
        abstract_geometry, 
        cl_coarse=config.cl_coarse, cl_fine=config.cl_fine, 
        verbose=False)

    fem_args = [
        paths['mesh'], paths['solution_dir'], paths['plastic_strain_dir'],
        str(config.b_1), str(config.y_1), str(config.b_2), str(config.y_2), 
        str(config.n_timesteps)
    ]
    
    
    fem_args = ' '.join(fem_args)
    command = f'{config.path_fem} {fem_args}'

    print(command)
    os.system(command)
