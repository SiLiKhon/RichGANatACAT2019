import os

import pandas as pd
import numpy as np
import tensorflow as tf

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

model_path = 'exported_model_tfScaler_attached'
model_name_format = "FastFastRICH_Cramer_{}_tfScaler"

particles = ['pion', 'kaon', 'proton', 'muon']
dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
feature_columns = [ 'Brunel_P', 'Brunel_ETA', 'nTracks_Brunel' ]

output_path = 'validation_output'

test_feature_values = dict(
    Brunel_P=10000., # MeV
    Brunel_ETA=3.,
    nTracks_Brunel=50
)

N_events = 10000

input_array = np.empty(dtype=np.float64, shape=(N_events, len(feature_columns)))
for i, feature in enumerate(feature_columns):
    input_array[:,i] = test_feature_values[feature]

for particle in particles:
    predictor = tf.contrib.predictor.from_saved_model(
        os.path.join(model_path, model_name_format.format(particle)),
        config=tf_config
    )

    predictions = predictor({'x' : input_array})['dlls']
    predictions = pd.DataFrame(predictions, columns=dll_columns)

    predictions.to_csv(os.path.join(output_path, '{}.csv'.format(particle)), index=False)

