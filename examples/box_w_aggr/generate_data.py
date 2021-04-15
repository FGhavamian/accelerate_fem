import os
import itertools

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
        'plastic_strain_dir': os.path.join(path_case_dir, 'plastic_strain/'),
        'vm_stress_dir': os.path.join(path_case_dir, 'vm_stress/')
    }

    os.makedirs(path_case_dir)
    os.makedirs(paths['solution_dir'])
    os.makedirs(paths['plastic_strain_dir'])
    os.makedirs(paths['vm_stress_dir'])

    return paths


def make_case_name(circle_location_seed, circle_radius_range):
    case_name = 'radius_'
    case_name += str(circle_radius_range[0])
    case_name += '_'
    case_name += str(circle_radius_range[1])
    case_name += '_'
    case_name += 'seed_'
    case_name += str(circle_location_seed)
    return case_name


def execute_one_case(circle_location_seed, circle_radius_range):
    case_name = make_case_name(circle_location_seed, circle_radius_range)

    paths = make_directories_for(case_name)

    abstract_geometry = AbstractGeometry(
        config.circle_density, 
        circle_radius_range, 
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
        paths['vm_stress_dir'],
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


if __name__ == '__main__':
    combinations = itertools.product(
        config.circle_location_seeds, 
        config.circle_radius_range)

    execute_one_case(*list(combinations)[0])

    # par = Parallel(n_jobs=-1, verbose=51)
    # delayed_func = delayed(execute_one_case)

    # par(
    #     delayed_func(loc, rad) 
    #     for loc, rad in combinations
    # )