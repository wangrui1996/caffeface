from __future__ import print_function
from __future__ import absolute_import

import caffe
import argparse
from caffe.model_libs import *
from config import generate_config, config, default
#from google.protobuf import text_format
import math
import os
import shutil
import stat
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Train face network")
    # general
    parser.add_argument("--network", default=default.network, help="network config")
    parser.add_argument("--loss", default=default.loss, help="loss config")
    parser.add_argument("--dataset", default=default.dataset, help="dataset")
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset, args.loss)
    args = parser.parse_args()
    return args

parse_args()

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.
caffe_root = os.getcwd()
# Modify the job name if you want.
job_name = "face_recognition"

dataset_name = "recognition"
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = False

#
copy_model = True

# If true, Remove old model files.
remove_old_models = False

train_txt = "data/{}/trainval.txt".format(dataset_name)
test_txt = "data/{}/test.txt".format(dataset_name)
#label_map_file = 'data/{}/label_map.txt'.format(dataset_name)

# Specify the batch sampler.
resize_width = 112
resize_height = 112
expand_ratio = 1.1
cutting_ratio = 0.3
prob = 0.7
batch_size = 1
resize = "{}x{}".format(resize_width, resize_height)
params_str = {
  'label_txt': train_txt,
  'batch_size': batch_size,
  'height': resize_height,
  'width': resize_width,
  'expand_ratio': expand_ratio,
  'cutting_ratio': cutting_ratio,
  'prob': prob,
  }


#params_str = "{\'label_txt\': \'{}\', \'batch_size\': ""{}, \'height\': {}, \'width\': {}, \'expand_ratio\': 1.1, \'cutting_ratio\': 0.3, \'prob\': 0.7,   \'train\': True,}".format(train_txt, batch_size, resize_height, resize_width)

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
use_batchnorm = False
lr_mult = 0.1
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.00004
base_lr = base_lr * 0.1



# The name of the model. Modify it if you want.
model_name = "{}_{}".format(job_name, dataset_name)

# Directory which stores the model .prototxt file.
save_dir = "models/{}/{}".format(job_name, dataset_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/{}/{}".format(job_name, dataset_name)
# Directory which stores the job script and log file.
job_dir = "jobs/{}/{}".format(job_name, dataset_name)

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)



#num_classes = len(open(label_map_file, 'r').readlines())
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.

# parameters for generating priors.
# minimum dimension of input image
min_dim = 48

# Solver parameters.
# Defining which GPUs to use.
gpus = "0"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.

accum_batch_size = 32
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])


# Evaluate on whole test set.
num_test_image = 1024
test_batch_size = 1
# Ideally test_batch_size should be divisible by num_test_image,
# otherwise mAP will be slightly off the true value.
test_iter = 1

solver_param = {
    # Train parameters
    'base_lr': 0.00001,
    'weight_decay': 0.0005,
    'lr_policy': "multistep",
    'stepvalue': [80000, 100000, 120000],
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': 4,
    'max_iter': 1200000,
    'snapshot': 10000,
    'display': 2,
    'average_loss': 50,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 50,
    'test_initialization': True,
    }


### Hopefully you don't need to change the following ###
# Check file.
check_if_exist(train_txt)
check_if_exist(test_txt)
#check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create deploy net.
net = caffe.NetSpec()
params_str['train'] = False
net.data, net.label = L.Data(ntop=2)
FmobileFaceNetBody(net, net.data, config)
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)
shutil.copy(deploy_net_file, job_dir)

# Create train net.
net = caffe.NetSpec()
params_str['train'] = True
net.data, net.label = L.Data(name="input_data",batch_size=default.per_batch_size, backend=P.Data.LMDB, source="data/face_recognition/lmdb",
                         include=dict(phase=caffe.TRAIN), transform_param=dict(), ntop=2)

body_layer = FmobileFaceNetBody(net, net.data, config)
fc_pre_class = L.InnerProduct(body_layer, num_output=config.num_classes)
net.loss = L.SoftmaxWithLoss(fc_pre_class, net.label)
net.softmax_layer = L.Softmax(fc_pre_class)
net.acc = L.Accuracy(net.softmax_layer, net.label)

with open(train_net_file, 'w') as f:
  print('name: "{}_train"'.format(model_name), file=f)
  print(net.to_proto(), file=f)
shutil.copy(train_net_file, job_dir)

# Create test net
net = caffe.NetSpec()

params_str['train'] = False

#net.data. net.label = L.Data(name="lmdbdata", source="/home/rui/lmdb", batch_size=1000, backend="LMDB")
net.data, net.label = L.DataPair(name = "input_data", batch_size=2, backend=P.Data.LMDB, source="/home/rui/data/tmplmdb",
                         include=dict(phase=caffe.TEST), transform_param=dict(), ntop=2)
body_layer = FmobileFaceNetBody(net, net.data, config)
#fc_pre = L.InnerProduct(body_layer, num_output=config.num_classes)
#net.softmax_layer = L.Softmax(fc_pre)


with open(test_net_file, 'w') as f:
    net_param = net.to_proto()
    del net_param.layer[-1]
    print('name: "{}_test"'.format(model_name), file=f)
    print(net_param, file=f)
shutil.copy(test_net_file, job_dir)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)
shutil.copy(solver_file, job_dir)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    print(basename)
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter
#train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
train_src_param = ""
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)


if copy_model:
  if max_iter > 0:
    train_src_param = '--weights="{}_iter_{}.caffemodel" \\\n'.format(snapshot_prefix, max_iter)


if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)
