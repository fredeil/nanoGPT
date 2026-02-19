# Experiment 2 - Width sweep
# Vary: n_embd ∈ {128, 192, 256, 384}
# Fixed: n_layer = 6, n_head = 8, dropout = 0.0

out_dir = 'out-exp2-width'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'tinystories'
wandb_run_name = 'exp2-width'

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# baby GPT model
n_layer = 6
n_head = 8
n_embd = 256  # overridden via command line
dropout = 0.0

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

device = 'mps'
compile = False
