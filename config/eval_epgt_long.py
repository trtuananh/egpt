import os
from .base_config import *
from . import eval_gpt2 as encoder_config

# -----------------------------------------------------------------------------
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

name = 'egpt_long' # name of the model, used for saving checkpoints
dataset = 'openwebtext_long' # use the smaller openwebtext dataset for faster training
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520

n_layer = 3
n_head = 12
batch_size = 4
block_size = 16
gradient_accumulation_steps = block_size

# this makes total number of tokens be 300B
resume_optimizer = True # if True, resume the optimizer state when training
learning_rate = 6e-4 # max learning rate
max_iters = 6000
warmup_iters = 20 # how many steps to warm up for
lr_decay_iters = 6000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 50
eval_iters = 50
eval_only = True
log_interval = 10

# weight decay
weight_decay = 1e-1

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_dict = {k: globals()[k] for k in config_keys} # will be useful for logging
config_dict["encoder_config"] = encoder_config.config_dict # add encoder config to the main config
# -----------------------------------------------------------------------------
