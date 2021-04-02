import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import accelerate_simulations.geometry as geom
import accelerate_simulations.preprocess as prep

import config


def load_model(path):
    return tf.keras.models.load_model(path)
    

def make_input_fields(circle_location_seed_list):
    input_fields_list = []
    for circle_location_seed in circle_location_seed_list:
        abstract_geometry = geom.AbstractGeometry(
            config.circle_density, 
            config.circle_radius_range, 
            config.box_size, 
            config.gap, 
            seed=circle_location_seed)

        rasterize = geom.GeometryRasterizer(resolution=config.resolution)
        x, y, raster_image = rasterize(abstract_geometry)

        make_geometric_fields = prep.GeometricFieldMaker(
            config.names_boundary, 
            geom.element_to_tag, 
            scaling_factor=3)

        input_fields = make_geometric_fields(raster_image, x, y)
        input_fields_list.append(input_fields)

    return np.stack(input_fields_list, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-model', dest='path_model')
    parser.add_argument('--circle-loc-seeds', dest='circle_loc_seeds',
                        help='separate integer seeds with comma')
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

    print('Average of one example:')
    print(f'preprocessing: {(t1-t0)/len(input_fields):.3f} seconds')
    print(f'prediction:    {(t3-t2)/len(pred):.3f} seconds')
    print(f'total:         {(t3-t2+t1-t0)/len(pred):.3f} seconds')

    for idx in range(len(circle_location_seed_list)):
        plt.figure()
        plt.imshow(np.squeeze(pred[idx]), cmap='jet')
        plt.colorbar()
        plt.savefig(f'outputs/seed_{circle_location_seed_list[idx]}.png')
        plt.close()
