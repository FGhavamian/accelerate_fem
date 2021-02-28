import os

from joblib import Parallel, delayed

from accelerate_simulations.geometry import AbstractGeometry, make_msh
import config


def make_directories_for(case_name):
    path_case_dir = os.path.join(config.path_data_raw, case_name)

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


def execute_one_case(circle_location_seed):
    case_name = str(circle_location_seed)

    paths = make_directories_for(case_name)

    abstract_geometry = AbstractGeometry(
        config.circle_density, 
        config.circle_radius_range, 
        config.box_size, 
        config.gap, 
        seed=circle_location_seed)

    abstract_geometry.save_at(paths['abstract_geometry'])

    make_msh(
        paths['mesh'], 
        abstract_geometry)

    fem_args = [
        paths['mesh'], 
        paths['solution_dir'], 
        paths['plastic_strain_dir'],
        str(config.b_1), 
        str(config.y_1), 
        str(config.b_2), 
        str(config.y_2), 
        str(config.n_timesteps),
        str(0)
    ]
    
    
    fem_args = ' '.join(fem_args)
    command = f'{config.path_fem} {fem_args}'

    print(command)
    os.system(command)


par = Parallel(n_jobs=-1, verbose=51)
delayed_func = delayed(execute_one_case)

par(
    delayed_func(circle_location_seed) 
    for circle_location_seed in config.circle_location_seed_list

)