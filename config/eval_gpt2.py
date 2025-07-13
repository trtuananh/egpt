# evaluate the base gpt2
n_layer=12, 
n_head=12, 
n_embd=768
# 124M parameters
batch_size = 8
block_size = 1024
eval_iters = 50 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'gpt2'

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config_dict = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
