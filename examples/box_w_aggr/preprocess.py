import os
import glob
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

import accelerate_simulations.geometry as geom
import accelerate_simulations.preprocess as prep
import config 


def read_abstract_geometry(path_example):
    path_geom = os.path.join(
        path_example,
        'abstract_geometry.pickle')

    with open(path_geom, 'rb') as file:
        return pickle.load(file)


def get_paths_examples():
    path_pattern = os.path.join(
        config.path_data_raw,
        '*'
    )
    return glob.glob(path_pattern)


def make_input_fields(path_example):
    abstract_geometry = read_abstract_geometry(path_example)
        
    rasterize = geom.GeometryRasterizer(resolution=config.resolution)
    raster_array = rasterize(abstract_geometry)
    
    make_material_fields = prep.MaterialFieldMaker(
        config.material_properties, 
        geom.element_to_tag)

    material_fields = make_material_fields(raster_array)

    make_geometric_fields = prep.GeometricFieldMaker(
        config.names_boundary, 
        geom.element_to_tag, 
        scaling_factor=1)

    geometric_fields = make_geometric_fields(raster_array)

    input_fields = np.concatenate(
        [geometric_fields, material_fields],
        axis=-1)

    return input_fields


def make_target_fields(path_example):
    target_fields = []
    for field_name in config.field_names:
        path_vtu = os.path.join(
            path_example, field_name, f'{field_name}_{config.time_step}.vtu')

        data_nodal = prep.read_vtu_file(path_vtu, field_name)

        target_field = prep.interpolate(data_nodal, config.box_size, config.resolution)
        target_fields.append(target_field)
    
    return np.concatenate(target_fields, axis=-1)


def save(input_fields, target_fields):
    if not os.path.exists(config.path_data_processed): 
        os.mkdir(config.path_data_processed)

    path_input = os.path.join(config.path_data_processed, 'input.npy')
    path_target = os.path.join(config.path_data_processed, 'target.npy')

    np.save(path_input, input_fields)
    np.save(path_target, target_fields)


def run_in_parallel(func, paths_examples):
    par = Parallel(n_jobs=-1, verbose=11)
    del_func = delayed(func)
    
    field_list = par(del_func(paths_example) for paths_example in paths_examples)
    return np.stack(field_list, axis=0) 


def remove_examples_with_unavailable_data(paths_examples):
    full_path = lambda p: os.path.join(
        p, 
        config.field_names[0], 
        f'{config.field_names[0]}_{config.time_step}.vtu'
    )
    return [p for p in paths_examples if os.path.exists(full_path(p))]


if __name__ == "__main__":
    paths_examples = get_paths_examples()
    
    print('[INFO] removing examples with unavailable data ...')
    print('[INFO] number of examples before:', len(paths_examples))
    paths_examples = remove_examples_with_unavailable_data(paths_examples)
    print('[INFO] number of examples before:', len(paths_examples))
    
    print('[INFO] making input fields ...')
    input_fields = run_in_parallel(make_input_fields, paths_examples)
    print('[INFO] input_fields shape:', input_fields.shape)

    print('[INFO] making target fields ...')
    target_fields = run_in_parallel(make_target_fields, paths_examples)
    print('[INFO] input_fields shape:', target_fields.shape)

    print('[INFO] saving fields ...')
    save(input_fields, target_fields)
