import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

from accelerate_simulations.train.model import UnetModel
import accelerate_simulations.geometry as geom
import accelerate_simulations.preprocess as prep


kernel_sizes = [(3, 3), (5, 5), (7, 7)]
n_circles = 5
circle_radius = 5
box_size = (100, 100)


def load_model(path):
    model = UnetModel(kernel_sizes)
    model.load_weights(path).expect_partial()
    return model


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


def make_input_fields(circle_location_seed_list):
    input_fields_list = []
    for circle_location_seed in circle_location_seed_list:
        abstract_geometry = geom.AbstractGeometry(
            n_circles,
            circle_radius,
            box_size,
            circle_location_seed)
        
        geometry_rasterizer = geom.GeometryRasterizer(abstract_geometry)
        geometry_rasterizer.rasterize()

        geometric_fields = make_geometric_fields(geometry_rasterizer)

        input_fields_list.append(geometric_fields)

    input_fields = np.stack(input_fields_list, axis=0)
    return input_fields


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-model', dest='path_model')
    parser.add_argument('--circle-loc-seeds', dest='circle_loc_seeds')
    args = vars(parser.parse_args())

    circle_location_seed_list = [
        int(s)
        for s in args['circle_loc_seeds'].split(',')]

    t0 = time.time()
    input_fields = make_input_fields(circle_location_seed_list)
    t1 = time.time()

    model = load_model(args['path_model'])

    t2 = time.time()
    pred = model.predict(input_fields)
    t3 = time.time()

    os.system('clear')
    print(f'data prepration time: {(t1-t0)/len(input_fields)}')
    print(f'prediction time: {(t3-t2)/len(pred)}')

    for idx in range(len(circle_location_seed_list)):
        plt.figure()
        plt.imshow(np.squeeze(pred[idx]))
        plt.colorbar()
        plt.savefig(f'outputs/seed_{circle_location_seed_list[idx]}.png')
