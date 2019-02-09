######### Imports & setup #########
import os
from collections import namedtuple

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib

import utils_rich_mrartemev as utils_rich

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0'

tf_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

models_path = "/home/martemev/data/Lambda/RICH-GAN/research/exported_model/"
preprocessors_path = "/home/martemev/data/Lambda/RICH-GAN/research/preprocessors/"

model_name_format = "FastFastRICH_Cramer_{}"
preprocessor_name_format = "FastFastRICH_Cramer_{}_preprocessor.pkl"

particles = ['kaon', 'pion', 'proton', 'muon']

output_filename = "predictions.pkl"

# We'll store splits in a single object
from collections import namedtuple
DataSplits = namedtuple("DataSplits", ['train', 'val', 'test'])



if __name__ == '__main__':

  ######### Loading the data and preparing utilities #########
  data_full = {
      particle : utils_rich.load_and_merge_and_cut(utils_rich.datasets[particle])
      for particle in particles
  }
  
  
  # Utility function to apply transfomations quickly
  def transform(arr, scaler, inverse=False):
      t = scaler.inverse_transform if inverse else scaler.transform
      XY_t = t(arr[:,:-1])
      W = arr[:,-1:]
      return np.concatenate([XY_t, W], axis=1)
  
  
  ######### Loading models and making predictions #########
  for particle in particles:
      with tf.Session(config=tf_config) as sess:
          predictor = tf.contrib.predictor.from_saved_model(
              os.path.join(models_path, model_name_format.format(particle))
          )
          scaler = joblib.load(
              os.path.join(preprocessors_path, preprocessor_name_format.format(particle))
          )
          
          data_full_t = transform(data_full[particle].values, scaler)
          
          predictions_t = predictor({'x' : data_full_t[:,utils_rich.y_count:]})['dlls']
          data_full_t[:,:utils_rich.y_count] = predictions_t
          data_full_predicted = transform(data_full_t, scaler, inverse=True)
          
          for i, col in enumerate(utils_rich.dll_columns):
              data_full[particle]["predicted_{}".format(col)] = data_full_predicted[:,i]
  
  splits = {
      particle : DataSplits(*utils_rich.split(data_full[particle]))
      for particle in particles
  }
  
  
  pd.to_pickle(splits, output_filename)
