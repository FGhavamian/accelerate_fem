import os
import argparse
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from sklearn.model_selection import train_test_split

from accelerate_simulations.train.model import (
    load_data,
    make_dataset,
    get_compiled_UnetModel
)
import config 
from util import *


parser = argparse.ArgumentParser()
parser.add_argument('--learning-rate', type=float)
parser.add_argument('--kernel-sizes', type=json.loads)
parser.add_argument('--filter-sizes', type=json.loads)
parser.add_argument('--epochs', type=int)
parser.add_argument('--model-name', type=str)
parser.add_argument('--initial-epoch', type=int, default=0)
parser.add_argument('--continue-training', type=int, default=0)
args = parser.parse_args()


print('[INFO] making directories to save model ...')
path_saved_models_dir = os.path.join(os.getcwd(), 'saved_models')
make_dir(path_saved_models_dir)

path_saved_model = os.path.join(path_saved_models_dir, args.model_name)


print('[INFO] tf version:', tf.__version__)
print('[INFO]', tf.config.list_physical_devices('GPU'))
print('[INFO] loading data ...')
inputs, targets = load_data(config.path_data_processed)

splits = train_test_split(inputs, targets, test_size=0.2, random_state=1)
inputs_train, inputs_test, targets_train, targets_test = splits


print('[INFO] training')
print('[INFO]   input  shape', inputs_train.shape)
print('[INFO]   target shape', targets_train.shape)
print('[INFO] test')
print('[INFO]   input  shape', inputs_test.shape)
print('[INFO]   target shape', targets_test.shape)
dataset_train = make_dataset(inputs_train, targets_train, config.batch_size, is_train=True)
dataset_test = make_dataset(inputs_test, targets_test, config.batch_size)

if args.continue_training == 1:
    print('[INFO] loading model ...')
    model = tf.keras.models.load_model(path_saved_model)
else:
    print('[INFO] building model ...')
    input_shape = (None,) + inputs_train.shape[1:]
    model = get_compiled_UnetModel(args.kernel_sizes, args.filter_sizes, input_shape)
    args.initial_epoch = 0


tf.keras.backend.set_value(model.optimizer.lr, args.learning_rate)


print(model.summary()) 
print('[INFO] fitting model ...')
history = model.fit(
    dataset_train, 
    epochs=args.epochs,
    validation_data=dataset_test,
    initial_epoch=args.initial_epoch
)

print(f'[INFO] saving model to {path_saved_model} ...')
model.save(path_saved_model)
print(f'[INFO] model saved to {path_saved_model}.')


path_metrics = os.path.join(path_saved_model, 'metrics.json')
print(f'[INFO] saving metrics to {path_metrics} ...')

metrics = history.history
metrics.update({'epochs': history.epoch})

if args.continue_training == 1:
    if os.path.exists(path_metrics):
        print(f'[INFO] updating metrics at {path_metrics} ...')
        metrics_old = load_json(path_metrics)
        metrics = update_metrics(metrics, metrics_old)

print(f'[INFO] saving metrics to {path_metrics} ...')
save_json(path_metrics, metrics)

