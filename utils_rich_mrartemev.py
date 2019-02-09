from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import numpy as np
import pandas as pd
import tensorflow as tf
import os

data_dir = '/home/martemev/data/Lambda/RICH-GAN/data/data_calibsample/'

def get_particle_dset(particle):
    return [data_dir + name for name in os.listdir(data_dir) if particle in name]

datasets = {particle: get_particle_dset(particle) for particle in ['kaon', 'pion', 'proton', 'muon', 'electron']} 


dll_columns = ['RichDLLe', 'RichDLLk', 'RichDLLmu', 'RichDLLp', 'RichDLLbt']
raw_feature_columns = [ 'Brunel_P', 'Brunel_ETA', 'nTracks_Brunel' ]
weight_col = 'probe_sWeight'
                     
y_count = len(dll_columns)
TEST_SIZE = 0.5

def load_and_cut(file_name):
    data = pd.read_csv(file_name, delimiter='\t')
    return data[dll_columns+raw_feature_columns+[weight_col]]

def load_and_merge_and_cut(filename_list):
    return pd.concat([load_and_cut(fname) for fname in filename_list], axis=0, ignore_index=True)

def split(data):
    data_train, data_val = train_test_split(data, test_size=TEST_SIZE, random_state=42)
    data_val, data_test = train_test_split(data_val, test_size=TEST_SIZE, random_state=1812)
    return data_train.reset_index(drop=True), \
           data_val  .reset_index(drop=True), \
           data_test .reset_index(drop=True)

def get_tf_dataset(dataset, batch_size):
    shuffler = tf.contrib.data.shuffle_and_repeat(dataset.shape[0])
    suffled_ds = shuffler(tf.data.Dataset.from_tensor_slices(dataset))
    return suffled_ds.batch(batch_size).prefetch(1).make_one_shot_iterator().get_next()

def scale_pandas(dataframe, scaler):
    return pd.DataFrame(scaler.transform(dataframe.values), columns=dataframe.columns)

def get_merged_typed_dataset(particle_type, dtype=None, log=False):
    file_list = datasets[particle_type]
    if log:
        print("Reading and concatenating datasets:")
        for fname in file_list: print("\t{}".format(fname))
    data_full = load_and_merge_and_cut(file_list)
    # Must split the whole to preserve train/test split""
    if log: print("splitting to train/val/test")
    data_train, data_val, _ = split(data_full)
    if log: print("fitting the scaler")
    print("scaler train sample size: {}".format(len(data_train)))
    scaler = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=int(1e5),
                                 subsample=int(1e10)).fit(data_train.drop(weight_col, axis=1).values)
    if log: print("scaling train set")
    data_train = pd.concat([scale_pandas(data_train.drop(weight_col, axis=1), scaler), data_train[weight_col]], axis=1)
    if log: print("scaling test set")
    data_val = pd.concat([scale_pandas(data_val.drop(weight_col, axis=1), scaler), data_val[weight_col]], axis=1)
    if dtype is not None:
        if log: print("converting dtype to {}".format(dtype))
        data_train = data_train.astype(dtype, copy=False)
        data_val = data_val.astype(dtype, copy=False)
    return data_train, data_val, scaler


def get_plots_hook(num_steps, ):
    class RunOpsEveryNSteps(tf.train.SessionRunHook):
        def __init__(self, num_steps, run_ops_fn, log=True, title=None):
            self._num_steps = num_steps
            self._run_ops_fn = run_ops_fn
            self._log = log
            if title is None:
                self._title = 'RunOpEvery_{}_Steps'.format(num_steps)
            else:
                self._title = title

        def begin(self):
            self._global_step_tensor = tf.train.get_or_create_global_step()
            self._run_ops = self._run_ops_fn()

        def after_run(self, run_context, run_values):
            gstep = run_context.session.run(self._global_step_tensor)
            if gstep % self._num_steps == 0:
                res = {k: run_context.session.run(v) for k, v in self._run_ops.items()}
                if self._log:
                    print("{}, Step #{}:".format(self._title, gstep))
                    for k, v in res.items():
                        print("  {} : {}".format(k, v))
                    
    def BuildEvalOp():
        def PlotGeneratedHist(x):
            fig = plt.figure(figsize=(10, 10))
            plt.hist(x, bins=int(np.ceil(len(x)**0.5)))
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)

            img = PIL.Image.open(buf)
            return np.array(img.getdata(), dtype=np.uint8).reshape(1, img.size[0], img.size[1], -1)

        fig = tf.py_func(PlotGeneratedHist, [preds_tensor], tf.uint8)
        return tf.summary.image("EvalImg", fig)
    return RunOpsEveryNSteps(num_steps, lambda: {'img_sum_op' : BuildEvalOp()}, log=False)
    
