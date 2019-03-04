import os

import tensorflow as tf
tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

from quantile_transformer_tf import QuantileTransformerTF


import utils_rich_mrartemev
import pandas as pd
import numpy as np

from sklearn.externals import joblib

from make_prediction import DataSplits

utils_rich = utils_rich_mrartemev
particles = ['kaon', 'pion', 'proton', 'muon']

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1'

data_full = {
    particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle])
    for particle in particles
}


models_path = "/home/martemev/data/Lambda/RICH-GAN/research/exported_model/"
preprocessors_path = "/home/martemev/data/Lambda/RICH-GAN/research/preprocessors/"

export_models_path = "exported_model_tfScaler_attached/"

model_name_format = "FastFastRICH_Cramer_{}"
preprocessor_name_format = "FastFastRICH_Cramer_{}_preprocessor.pkl"


def tf_scalers_from_scaler(scaler, dtype=np.float64):
    scaler_x_tf = QuantileTransformerTF(scaler,
                                        list(range(utils_rich.y_count, scaler.quantiles_.shape[1])),
                                        dtype)
    scaler_y_tf = QuantileTransformerTF(scaler,
                                        list(range(0, utils_rich.y_count)),
                                        dtype)
    return scaler_x_tf, scaler_y_tf

for particle in particles:
    print("Working on {}s".format(particle))
    predictor = tf.contrib.predictor.from_saved_model(
                    os.path.join(models_path, model_name_format.format(particle)),
                    config=tf_config
                )
    print("Loaded predictor")

    scaler = joblib.load(
        os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
    )
    print("Loaded scaler")
    
    input_name, = [x.name for x in predictor.feed_tensors['x'].consumers()[0].outputs]
    output_name = predictor.fetch_tensors['dlls'].name
    
    mod_graph = tf.Graph()
    with mod_graph.as_default():
        tf_scaler_x, tf_scaler_y = tf_scalers_from_scaler(scaler)
        print("Created tf scalers")
        
        input_tensor = tf.placeholder(dtype=tf.float64, shape=(None, len(utils_rich.raw_feature_columns)), name='x')
        scaled_input = tf.cast(tf_scaler_x.transform(input_tensor, False), dtype=tf.float32)
        
        with tf.Session(config=tf_config) as sess:
            meta_graph_def = tf.saved_model.loader.load(
                sess, ['serve'],
                os.path.join(models_path, model_name_format.format(particle)),
                input_map={input_name : scaled_input}
            )
            print("Reloaded the model with input_map")

            scaled_output = mod_graph.get_tensor_by_name(output_name)
            output_tensor = tf_scaler_y.transform(tf.cast(scaled_output, dtype=tf.float64), True)

            sub_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, [output_tensor.op.name]
            )
            print("Sub-graph created")
    
    reduced_graph = tf.Graph()
    with reduced_graph.as_default():
        with tf.Session(config=tf_config) as sess:
            tf.import_graph_def(sub_graph_def, name='')
            input_tensor  = reduced_graph.get_tensor_by_name(input_tensor.name)
            output_tensor = reduced_graph.get_tensor_by_name(output_tensor.name)

            predictions = sess.run(
                output_tensor,
                feed_dict={
                    input_tensor : data_full[particle][utils_rich.raw_feature_columns].values
                }
            )
            print("Calculated predictions with the sub-graph")
            
            for i, col in enumerate(utils_rich.dll_columns):
                data_full[particle]["predicted_{}".format(col)] = predictions[:,i]

            model_export_dir = os.path.join(export_models_path, model_name_format.format(particle) + "_tfScaler")
            tf.saved_model.simple_save(
                sess, model_export_dir,
                inputs={"x": input_tensor},
                outputs={"dlls": output_tensor}
            )
            print("Exported the sub-graph model")



splits = {
    particle : DataSplits(*utils_rich.split(data_full[particle]))
    for particle in particles
}

pd.to_pickle(splits, 'predictions_with_tfScaler.pkl')
print("Saved predictions to pickle")
