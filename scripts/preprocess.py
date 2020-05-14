import os
import glob
import pickle
import argparse

import numpy as np

import accelerate_simulations.geometry as geom
import accelerate_simulations.preprocess as prep


def read_abstract_geometry(path_example):
    path_geom = os.path.join(
        path_example,
        'abstract_geometry.pickle')

    with open(path_geom, 'rb') as file:
        return pickle.load(file)


def get_paths_examples(path_data):
    path_pattern = os.path.join(
        path_data,
        '*'
    )
    return glob.glob(path_pattern)


def make_geometric_fields(geometry_rasterizer):
    names_boundary = [
            'boundary_left',
            'boundary_bot',
            'boundary_right',
            'boundary_top',
            'circle_boundary']

    make_geometric_fields = prep.GeometricFieldMaker(
        geometry_rasterizer,
        names_boundary,
        scaling_factor=1)

    return make_geometric_fields()


def make_material_fields(geometry_rasterizer):
    material_properties = {
        'y': {'box_wo_holes': 10, 'holes': 1},
        'b': {'box_wo_holes': 100, 'holes': 100}
    }

    make_material_fields = prep.MaterialFieldMaker(
        geometry_rasterizer,
        material_properties)

    return make_material_fields()


def make_input_fields(path_examples):
    input_fields_list = []
    for path_example in paths_examples:
        abstract_geometry = read_abstract_geometry(path_example)
        geometry_rasterizer = geom.GeometryRasterizer(abstract_geometry)
        geometry_rasterizer.rasterize()

        geometric_fields = make_geometric_fields(geometry_rasterizer)
        material_fields = make_material_fields(geometry_rasterizer)

        input_fields = np.concatenate(
            [geometric_fields, material_fields],
            axis=-1)

        input_fields_list.append(input_fields)

    input_fields = np.stack(input_fields_list, axis=0) 
    return input_fields


def make_target_fields(paths_examples):
    field_name = 'plastic_strain'
    box_coords = ((0, 0), (100, 100))
    grid_x, grid_y, grid_flat = prep.make_grid(box_coords, 1)

    target_fields = []
    for path_example in paths_examples:
        path_vtu = os.path.join(
            path_example, field_name, f'{field_name}_50.vtu')

        data_nodal = prep.read_vtu_file(path_vtu, field_name)

        target_field = prep.interpolate(data_nodal, grid_flat, grid_y.shape)
        target_fields.append(target_field)

    target_fields = np.stack(target_fields, axis=0)
    target_fields = np.expand_dims(target_fields, axis=-1)

    return target_fields


def save(input_fields, target_fields, path_processed):
    path_input = os.path.join(path_processed, 'input.npy')
    path_target = os.path.join(path_processed, 'target.npy')

    np.save(path_input, input_fields)
    np.save(path_target, target_fields)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-data', dest='path_data')
    args = vars(parser.parse_args())

    paths_examples = get_paths_examples(args['path_data'])

    print('[INFO] making input fields ...')
    input_fields = make_input_fields(paths_examples)
    print('[INFO] input_fields shape:', input_fields.shape)

    print('[INFO] making target fields ...')
    target_fields = make_target_fields(paths_examples)
    print('[INFO] input_fields shape:', target_fields.shape)

    print('[INFO] saving fields ...')
    paths_processed = args['path_data'].replace('raw', 'processed')
    save(input_fields, target_fields, paths_processed)
