"""
A variation from nanoGPT: A much shorter version of train.py for benchmarking

The benchmark is from here: https://huggingface.co/docs/transformers/perplexity
"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from model import GPTConfig, GPT
from tqdm import trange
#from torcheval.metrics import Perplexity

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
n_layer = 32
n_head = 16
n_embd = 256
dropout = 0.0
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init
if real_data:

    dataset = 'tinystories'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    def get_batch():
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout)


print(f"Resuming training from {out_dir}")
# resume training from a checkpoint.
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
# fix the keys of the state dictionary :(
# honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0
model = model.to(device)
# simple benchmarking
nlls = []

torch.cuda.synchronize()

for num_steps in trange(100): 
    X, Y = get_batch()
    #print(X.device.type, Y.device.type) 
    with torch.no_grad():
        logits, loss = model(X, Y)
        neg_log_likelihood = loss
    nlls.append(neg_log_likelihood.detach().cpu())
    torch.cuda.synchronize()
   
print(f"Average negative lock likelihood over 100 steps: {torch.stack(nlls).mean()}")
perplexity = torch.exp(torch.stack(nlls).mean())
print(f"Perplexity: {perplexity.item()}")
