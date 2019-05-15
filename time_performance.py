import os
import time
import pickle

import numpy as np
import tensorflow as tf
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                           intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)
from tqdm import tqdm

import utils_rich_mrartemev
from make_prediction import DataSplits

utils_rich = utils_rich_mrartemev
particles = ['kaon', 'kaon_full', 'pion', 'proton', 'muon']

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''


models_path_raw = "/home/amaevskiy/data/RICH-GAN/martemev_models/exported_model/"
models_path_scaler = "exported_model_tfScaler_attached_v3"
model_name_format = "FastFastRICH_Cramer_{}"

batch_sizes = np.logspace(0, 4, 5, dtype=int)
TOTAL_EVENTS = batch_sizes[-1]

output_file = 'timings.pkl'

data_full = {
    particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle.replace('_full', '')])
    for particle in particles
}

def time_performance(predictor, data, batch_size=None):
    if batch_size is None:
        batch_size = len(data)
    data = [data[i : i + batch_size].copy() for i in range(0, len(data), batch_size)]
    timings = np.empty(shape=len(data), dtype=np.float64)

    for i, batch in tqdm(enumerate(data), total=len(data)):
        start_time = time.perf_counter()
        predictions = predictor({'x' : batch})['dlls']
        timings[i] = time.perf_counter() - start_time
        timings[i] = len(batch) / timings[i]

    return timings

results = dict(raw={}, scaler={}) 

for particle in particles:
    print("Working on {}s".format(particle))
    predictor_raw = tf.contrib.predictor.from_saved_model(
                        os.path.join(models_path_raw, model_name_format.format(particle)),
                        config=tf_config
                    )
    predictor_scaler = tf.contrib.predictor.from_saved_model(
                           os.path.join(models_path_scaler, model_name_format.format(particle) + '_tfScaler'),
                           config=tf_config
                       )
    print("  Loaded predictors")

    results['raw'   ][particle] = {}
    results['scaler'][particle] = {}

    sample_raw    = data_full[particle][utils_rich.raw_feature_columns + [utils_rich.weight_col]].values[:TOTAL_EVENTS]
    sample_scaler = data_full[particle][utils_rich.raw_feature_columns].values[:TOTAL_EVENTS]

    for batch_size in batch_sizes:
        print("  batch size:", batch_size)
        print("    without scaler...")
        results['raw'   ][particle][batch_size] = time_performance(predictor_raw   , sample_raw   , batch_size)
        print("    with scaler...")
        results['scaler'][particle][batch_size] = time_performance(predictor_scaler, sample_scaler, batch_size)


with open(output_file, 'wb') as f:
    pickle.dump(results, f)

print("done")
