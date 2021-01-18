import os 
import json 

import pandas as pd


def make_dir(path):
    if not os.path.exists(path): 
        os.mkdir(path)


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def update_metrics(metrics, metrics_old):
    metrics_final = metrics.copy()

    metric_keys = [k for k in metrics.keys() if k!='epochs']

    for i, epoch in enumerate(metrics_old['epochs']):
        if epoch not in metrics_final['epochs']:
            metrics_final['epochs'].append(epoch)
            for metric_key in metric_keys:
                metrics_final[metric_key].append(metrics_old[metric_key][i])

    return metrics_final


    
if __name__ == "__main__":
    metrics_old = {
        'epochs': [1,2,3,4,5,6],
        'v': [11,12,13,14,15,16],
        'v2': [110,120,103,104,105,106]
    }

    metrics = {
        'epochs': [4,5,6,7],
        'v': [14,15,16,19],
        'v2': [104,105,106,19999]
    }

    metrics = update_metrics(metrics, metrics_old)

    print(metrics)