import os
from .base_config import *
from . import eval_gpt2 as encoder_config

# -----------------------------------------------------------------------------
# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

out_dir = os.path.join('out', 'epgt')
wandb_log = True
wandb_project = 'owt'
wandb_run_name='gpt2-124M'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520

n_layer = 3
n_head = 12
batch_size = 1
block_size = 16
gradient_accumulation_steps = block_size

# this makes total number of tokens be 300B
max_iters = 5000
lr_decay_iters = 5000

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
