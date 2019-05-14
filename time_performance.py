import os
import time

import tensorflow as tf
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                           intra_op_parallelism_threads=1,
                           inter_op_parallelism_threads=1)

import utils_rich_mrartemev
from make_prediction import DataSplits

utils_rich = utils_rich_mrartemev
particles = ['kaon', 'kaon_full', 'pion', 'proton', 'muon']

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=''


models_path_raw = "/home/amaevskiy/data/RICH-GAN/martemev_models/exported_model/"
models_path_scaler = "exported_model_tfScaler_attached_v3"
model_name_format = "FastFastRICH_Cramer_{}"

data_full = {
    particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle.replace('_full', '')])
    for particle in particles
}


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
    print("Loaded predictors")

    sample = data_full[particle][utils_rich.raw_feature_columns + [utils_rich.weight_col]].values
    start_t = time.perf_counter()
    predictions_raw = predictor_raw({'x' : sample})['dlls']
    time_raw = time.perf_counter() - start_t

    sample = data_full[particle][utils_rich.raw_feature_columns].values
    start_t = time.perf_counter()
    predictions_scaler = predictor_scaler({'x' : sample})['dlls']
    time_scaler = time.perf_counter() - start_t

    print("Time without scaler:", time_raw   , 'secs')
    print("Time with scaler:   ", time_scaler, 'secs')
    print("")
