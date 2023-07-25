"""Params for ADDA."""
import os
from datetime import datetime
from myfunc import set_determinism, set_logger
import argparse

parser = argparse.ArgumentParser(description='choose config')
parser.add_argument('--parent', type=str, required=True)
parser.add_argument('--dset', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
args = parser.parse_args()

parent = args.parent
dset = args.dset
task = args.task

fulld0, fulld1 = dset.split('_')

set_determinism()

# params for dataset and data loader
data_root = "/nas/data/syamagami/GDA/data"
text_root = f"data/{parent}/{task}/{dset}"
labeled_filename = os.path.join(text_root, 'labeled.txt')
unlabeled_filename = os.path.join(text_root, 'unlabeled.txt')
test0_filename = os.path.join(text_root, f'test_{fulld0}.txt')
test1_filename = os.path.join(text_root, f'test_{fulld1}.txt')
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
# batch_size = 50
batch_size = 12
image_size = 64

if parent == 'Office31':
    num_classes = 31
elif parent == 'OfficeHome':
    num_classes = 65
else:
    raise ValueError(f'parent: {parent}')

# log
now = datetime.now().strftime("%y%m%d_%H:%M:%S")
cuda = ''.join([str(i) for i in os.environ['CUDA_VISIBLE_DEVICES']])
exec_num = os.environ['exec_num'] if 'exec_num' in os.environ.keys() else 0
log_dir  =  f'logs/{parent}/{now}--c{cuda}n{exec_num}--{dset}--{task}'
set_logger(log_dir)

# params for source dataset
src_dataset = parent
src_domain = dset.split('_')[0]
src_encoder_restore = "snapshots/ADDA-source-encoder-final.pt"
src_classifier_restore = "snapshots/ADDA-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_dataset = parent
tgt_domain = dset.split('_')[1]
tgt_encoder_restore = "snapshots/ADDA-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "snapshots"
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2
d_model_restore = "snapshots/ADDA-critic-final.pt"

# params for training network
num_gpu = 1
num_epochs_pre = 100
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
num_epochs = 2000
log_step = 100
log_per_epoch = 10
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
