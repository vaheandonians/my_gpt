"""
TRAINING SCRIPT
===============

This script trains a Large Language Model (LLM). 
It is designed to explain the "magic" terms you often hear in AI:
DDP, Warmup, Gradient Accumulation, Mixed Precision, etc.
"""

import os
import time
import math
from contextlib import nullcontext
from dataclasses import asdict

import torch
# DDP = Distributed Data Parallel. This is PyTorch's way of training on multiple GPUs.
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
from datasets import load_dataset

from model import Config, GPT

# ==============================================================================
#   1. CONCEPT: THE "HYPERPARAMETERS"
#   These are the knobs and dials we turn before training starts.
# ==============================================================================

# --- I/O Settings ---
out_dir = 'out_fineweb'
eval_interval = 500               # We don't want to check accuracy every step (too slow).
sample_interval = 100             # Occasionally let the model "speak" to see if it's making sense.
log_interval = 10                 
eval_iters = 20                   # When we check accuracy, we test on 20 random batches to get a stable average.
always_save_checkpoint = False    

# --- Hardware Logic ---
# 1. CUDA: The standard. NVIDIA GPUs.
# 2. MPS: "Metal Performance Shaders". This allows Macs (M1/M2/M3) to train using their GPU cores.
# 3. CPU: The Central Processing Unit. It does math 100x slower than a GPU. Only for debugging.
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

# --- Dataset Settings ---
# We use a Parquet file. Parquet is a compressed column-storage format (like a super-efficient Excel).
dataset_path = "training_data/fineweb_edu_sample/sample/10BT/000_00000.parquet"

# --- CONCEPT: BATCH SIZE vs. GPU MEMORY ---
# The Batch Size is how many separate documents the GPU reads in parallel.
# Ideally, we want this HUGE (e.g., 500) so the model learns general rules, not outliers.
# However, GPU memory (VRAM) is limited (e.g., 24GB).
# If we try to fit 500 documents, the GPU crashes (Out of Memory).
# So, we pick a small number that fits physically in the hardware (e.g., 12).
batch_size = 12

block_size = 384 # The "Context Window". The model can see 1024 tokens back in time.

# --- CONCEPT: GRADIENT ACCUMULATION ---
# Problem: We established that `batch_size=12` is too small for stable learning.
#          Noisy gradients will make the model learning erratic.
# Solution: "Gradient Accumulation".
# We pretend we have a huge GPU. 
# 1. We run the model on 12 samples. Calculate errors. DO NOT update weights yet.
# 2. We run the model on another 12 samples. Add these errors to the previous ones.
# 3. Repeat this 40 times.
# 4. NOW update the weights.
# Effective Batch Size = 12 * 40 = 480. (Stable learning on small hardware!)
gradient_accumulation_steps = 40  

# --- Optimizer Settings ---
learning_rate = 6e-4              # How big of a step we take towards the solution.
max_iters = 100000                # How long we train.

# --- CONCEPT: WEIGHT DECAY ---
# Imagine the model is lazy. It might try to solve the problem by making one weight HUGE 
# and ignoring everything else. This is "Overfitting".
# Weight Decay is a penalty that says: "You can solve the problem, but try to keep
# your numbers (weights) close to zero." This forces the model to use ALL its neurons,
# resulting in a smarter, more general brain.
weight_decay = 1e-1               

beta1 = 0.9                       
beta2 = 0.95
grad_clip = 1.0                   # If the model panics and suggests a massive change, cap it at 1.0.

# --- CONCEPT: WARMUP & DECAY (SCHEDULER) ---
# When training starts, the model's brain is random noise.
# If we try to learn at full speed (max learning_rate) immediately, the model might
# take a huge step in a random direction and "break" (diverge).
#
# Phase 1: WARMUP. Start with tiny steps. Slowly accelerate over 2000 steps.
#          This lets the model find a stable footing.
# Phase 2: COSINE DECAY. As the model gets smarter, we should take smaller steps.
#          Think of golf: You swing hard at the start (drive), but tap gently 
#          near the hole (putt). If you swing hard near the hole, you miss.
decay_lr = True                   
warmup_iters = 2000               
lr_decay_iters = 100000           
min_lr = 6e-5                     

# ==============================================================================
#   2. CONCEPT: DISTRIBUTED DATA PARALLEL (DDP)
#   "How to train on 8 GPUs at once"
# ==============================================================================

# How does the script know if it's running on just one laptop or a massive cluster?
# We check environment variables. If 'RANK' is set, we are in a cluster.
ddp = int(os.environ.get('RANK', -1)) != -1 

if ddp:
    # We use 'nccl' (NVIDIA Collective Communications Library).
    # It's a super-fast language for GPUs to talk to each other over ethernet/infiniband.
    init_process_group(backend='nccl')

    # --- CONCEPT: RANK vs LOCAL_RANK vs WORLD_SIZE ---
    # Imagine a classroom of 8 students (GPUs) taking a test.
    # WORLD_SIZE = 8. (The total number of students).
    # RANK       = The Student ID (0 to 7). Rank 0 is usually the "Class President".
    # LOCAL_RANK = Which desk in the specific room they are sitting at.
    #              (If you have 2 computers with 4 GPUs each: 
    #               Computer A has Ranks 0,1,2,3. Local Ranks 0,1,2,3.
    #               Computer B has Ranks 4,5,6,7. Local Ranks 0,1,2,3.)
    
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    # Set the GPU for this specific process
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    # Only Rank 0 (the boss) is allowed to print to the screen or save files.
    # Otherwise, you get 8 copies of the same print statement messing up your terminal.
    master_process = ddp_rank == 0 
    
    # We shift the random seed so each GPU gets DIFFERENT random data.
    # If they all got the same data, they would learn the exact same thing, 
    # defeating the purpose of parallel training!
    seed_offset = ddp_rank 
    
    # If we have 8 GPUs, we don't need to accumulate as much on each individual GPU
    # to hit our target global batch size. We split the workload.
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # Vanilla settings for single GPU
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_local_rank = 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"Tokens per iteration: {gradient_accumulation_steps * ddp_world_size * batch_size * block_size:,}")
    print(f"Using device: {device}")

torch.manual_seed(1337 + seed_offset)

# Enable TF32. This is a special math mode on newer NVIDIA cards (Ampere/Hopper).
# It does matrix multiplication with slightly less precision (19 bits vs 23 bits)
# but runs much faster. The AI doesn't notice the difference.
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True

# ==============================================================================
#   3. CONCEPT: MIXED PRECISION & BFLOAT16
# ==============================================================================
# Computers usually do math in "Float32" (32 bits, lots of decimal places).
# AI models are fuzzy; they don't need 10 decimal places.
# "Float16" uses half the memory and runs faster.
# "BFloat16" (Brain Float) is a Google invention that keeps the "Range" of 32-bit
# but sacrifices the tiny details. It is the gold standard for LLM training now.

device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')

if device_type == 'cuda':
    # Use bfloat16 if the hardware supports it, otherwise fallback to float16
    dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
elif device_type == 'mps':
    dtype = 'bfloat16' # Macs love bfloat16
else:
    dtype = 'float32'

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# The Context Manager ("ctx")
# When we enter "with ctx:", PyTorch automatically casts big matrices down to
# the smaller dtype (bfloat16) to save memory/time.
if device_type == 'cuda':
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
elif device_type == 'mps':
    ctx = torch.amp.autocast(device_type='mps', dtype=ptdtype)
else:
    ctx = nullcontext()

# ==============================================================================
#   4. DATA LOADER: STREAMING & STACKING
# ==============================================================================

print(f"Loading dataset from {dataset_path}...")
try:
    hf_dataset = load_dataset("parquet", data_files=dataset_path, split="train")
    enc = tiktoken.get_encoding("gpt2")
except Exception as e:
    print(f"Error: Could not load dataset or tokenizer. {e}")
    exit()

def get_batch(split):
    """
    This function prepares a 'brick' of data to feed the GPU.
    The GPU expects a perfect rectangle of numbers: (Batch_Size, Block_Size).
    """
    data = hf_dataset 
    
    # 1. Randomly pick 'batch_size' number of documents index.
    ix = torch.randint(len(data), (batch_size,))
    
    # --- CONCEPT: X_STACK and Y_STACK ---
    # We are building a list of rows.
    # X is what the model SEES.
    # Y is what the model should PREDICT.
    # Example:
    # Text: "The cat sat on"
    # X (Input):  "The cat sat"
    # Y (Target): "cat sat on"
    # (The model sees "The", predicts "cat". Sees "cat", predicts "sat".)
    x_stack = []
    y_stack = []

    for i in ix:
        text = data[int(i)]['text'] 
        tokens = enc.encode(text)
        
        # If the web page is too short (e.g. just "Error 404"), we can't learn from it.
        # We grab a new random doc and append it until we have enough text.
        while len(tokens) <= block_size + 1:
            random_idx = torch.randint(len(data), (1,)).item()
            new_text = data[random_idx]['text']
            tokens += enc.encode(new_text)

        # Random Crop: We grab a random chunk of 1024 tokens from the document.
        max_start = len(tokens) - block_size - 1
        start_idx = torch.randint(max_start, (1,)).item()
        
        # Slicing the tokens
        x_chunk = torch.tensor(tokens[start_idx : start_idx + block_size], dtype=torch.long)
        y_chunk = torch.tensor(tokens[start_idx+1 : start_idx + block_size + 1], dtype=torch.long)
        
        x_stack.append(x_chunk)
        y_stack.append(y_chunk)

    # Convert the list of rows into a PyTorch Tensor (The "Brick")
    x = torch.stack(x_stack)
    y = torch.stack(y_stack)

    # Move to GPU
    if device_type == 'cuda':
        # pin_memory acts as a fast lane between CPU RAM and GPU VRAM.
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
        
    return x, y

# ==============================================================================
#   5. MODEL & OPTIMIZER
# ==============================================================================

config = Config(block_size=block_size, vocab_size=enc.n_vocab)

if master_process:
    print("Initializing model from scratch...")

model = GPT(config)
model.to(device)

# --- CONCEPT: GRAD SCALER ---
# In Float16, numbers can get REALLY small (e.g., 0.00000001).
# Computer says: "That's basically zero." -> Gradient vanishes. Model stops learning.
# The Scaler multiplies the loss by 65,536 (scales it up).
# We do the math in the safe range.
# Then we divide by 65,536 before updating weights.
enable_scaler = (dtype == 'float16') and (device_type == 'cuda')
scaler = torch.amp.GradScaler('cuda', enabled=enable_scaler)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

# --- CONCEPT: TORCH COMPILE ---
# This is "Just-In-Time" (JIT) compilation.
# Python is an interpreted language (slow).
# torch.compile looks at your GPT model and converts it into a single, optimized 
# C++ / CUDA kernel. It can make training 30% faster.
if device_type == 'cuda':
    print("Compiling model... (This may take a minute)")
    model = torch.compile(model, backend="eager")
elif device_type == 'mps':
    print("Running on MPS: Skipping torch.compile (not fully stable yet)")

# Wrap model in DDP container
if ddp:
    # This wrapper handles the communication.
    # When we do backward(), DDP automatically sums gradients across all GPUs.
    model = DDP(model, device_ids=[ddp_local_rank])

# ==============================================================================
#   6. HELPERS
# ==============================================================================

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # Turn off dropout (we want deterministic results for evaluation)
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('train')
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    out['train'] = losses.mean()
    out['val'] = losses.mean()
    model.train() # Turn dropout back on for training
    return out

@torch.no_grad()
def generate_sample(prompt="The meaning of life is", max_tokens=50):
    # This generates text so we can visually see if the model is learning English.
    model.eval()
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    for _ in range(max_tokens):
        # Crop to the last block_size tokens so we don't exceed context window
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond)
        # Take the logits for the LAST token in the sequence
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        # Sample from the distribution (Concept: Multinomial Sampling)
        # We don't just pick the highest probability (greedy). We roll the dice
        # weighted by probability. This makes the text less robotic.
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
    
    output = enc.decode(x[0].tolist())
    model.train()
    return output

def get_lr(it):
    # Returns the learning rate for step 'it' based on warmup and cosine decay.
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ==============================================================================
#   7. TRAINING LOOP (THE MAIN EVENT)
# ==============================================================================

iter_num = 0
best_val_loss = 1e9
t0 = time.time()
raw_model = model.module if ddp else model

print("Starting training loop...")

while True:
    # --- A. Set Learning Rate ---
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # --- B. Evaluation & Logging ---
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Save if it's the best model so far
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'config': asdict(config),
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # Generate a text sample to check progress
    if iter_num % sample_interval == 0 and master_process:
        sample = generate_sample()
        print(f"\n--- Sample at step {iter_num} ---\n{sample}\n{'-'*40}")

    # --- C. Forward & Backward Loop (Gradient Accumulation) ---
    for micro_step in range(gradient_accumulation_steps):
        # DDP Sync Logic:
        # If we are on the LAST micro_step, we enable synchronization.
        # This tells DDP: "Okay, we are done accumulating. Now talk to the other GPUs 
        # and average our gradients together."
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            
        X, Y = get_batch('train')
        
        # Mixed Precision Context
        with ctx: 
            logits, loss = model(X, Y)
            # IMPORTANT: We divide the loss.
            # If we sum gradients from 40 steps, the gradient will be 40x bigger than normal.
            # Dividing by 40 keeps the scale correct (Average).
            loss = loss / gradient_accumulation_steps 
        
        # Backward Pass
        # This calculates "How much should we change every weight?" (Gradients)
        scaler.scale(loss).backward()
        
    # --- D. Optimizer Step ---
    # Gradient Clipping
    # Sometimes gradients explode to infinity. We clip them to length 1.0 to stay safe.
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
    # Apply the changes to the weights
    scaler.step(optimizer)
    scaler.update() 
    
    # Flush Gradients
    # set_to_none=True is a PyTorch trick. It's faster than setting to 0 because 
    # it avoids writing a memory block full of zeros. It just deletes the pointer.
    optimizer.zero_grad(set_to_none=True)

    # --- E. Logging Stats ---
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.4e}")

    iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()