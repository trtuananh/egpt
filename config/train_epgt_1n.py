import os
from .base_config import *
from . import eval_gpt2 as encoder_config

# -----------------------------------------------------------------------------
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
out_dir = os.path.join('out', 'epgt_1n')
wandb_log = False
wandb_project = 'owt'
wandb_run_name='epgt_1n'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520

n_layer = 3
n_head = 12
batch_size = 4
block_size = 1
gradient_accumulation_steps = block_size

# this makes total number of tokens be 300B
resume_optimizer = True # if True, reset the optimizer state when resuming training
learning_rate = 6e-4 # max learning rate
max_iters = 600
warmup_iters = 2 # how many steps to warm up for
decay_lr = False # whether to decay the learning rate
lr_decay_iters = 600 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 3
eval_iters = 1
log_interval = 1

# weight decay
weight_decay = 1e-1

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_dict = {k: globals()[k] for k in config_keys} # will be useful for logging
config_dict["encoder_config"] = encoder_config.config_dict # add encoder config to the main config
# -----------------------------------------------------------------------------
