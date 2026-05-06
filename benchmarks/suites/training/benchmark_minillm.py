#!/usr/bin/env python3
"""
Benchmark: SMO variants vs AdamW on a Mini-LLM (GPT-style Transformer).
Tests the hypothesis that dense Linear weights in Transformers can be treated 
as "textures" and spatially/spectrally compressed without destroying attention logic.
"""

import argparse
import math
import sys
import time
import random
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks._paths import add_project_root_to_path
add_project_root_to_path()

# Benchmark classification: family=end_to_end_training, category=smoke, status=canonical
import torch
import torch.nn as nn
from torch.nn import functional as F
from smo import SMO
from smo.optimizers.spatial_8bit import SMO8bit
from benchmarks.results_utils import make_run_record, write_benchmark_bundle


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 1. Mini-LLM Architecture (NanoGPT style)
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTConfig:
    vocab_size = 65
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 128
    bias = False

class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_optimizer_memory(optimizer):
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
    return total / (1024 ** 2)

# Dummy DataLoader for character-level data
def get_batch(split, data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)

# -----------------------------------------------------------------------------
# 3. Experiment Runner
# -----------------------------------------------------------------------------
def run_experiment(optimizer_name, optimizer_fn, max_iters=500, device='cpu', seed=1234):
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Running: {optimizer_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Synthetic data generation (random tokens)
    vocab_size = GPTConfig.vocab_size
    train_data = torch.randint(0, vocab_size, (10000,))
    val_data = torch.randint(0, vocab_size, (1000,))
    
    config = GPTConfig()
    model = MiniGPT(config).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")
    
    optimizer = optimizer_fn(model.parameters())
    
    # Dummy step to initialize states
    xb, yb = get_batch('train', train_data, config.block_size, 4, device)
    _, loss = model(xb, yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    opt_mem_before = get_optimizer_memory(optimizer)
    print(f"Optimizer Memory (Initial): {opt_mem_before:.2f} MB")
    
    results = {
        'optimizer': optimizer_name,
        'parameters': param_count,
        'iters': [],
        'train_time': 0,
        'seed': seed,
    }
    
    model.train()
    start_time = time.time()
    
    # Training Loop
    for iter_num in range(1, max_iters + 1):
        xb, yb = get_batch('train', train_data, config.block_size, 16, device)
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_num % 100 == 0 or iter_num == max_iters:
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_xb, val_yb = get_batch('val', val_data, config.block_size, 16, device)
                _, val_loss = model(val_xb, val_yb)
                val_perplexity = math.exp(val_loss.item())
            model.train()
            
            elapsed = time.time() - start_time
            print(f"Iter {iter_num}/{max_iters} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val PPL: {val_perplexity:.2f} | Time: {elapsed:.2f}s")
            
            results['iters'].append({
                'iter': iter_num,
                'train_loss': round(loss.item(), 4),
                'val_loss': round(val_loss.item(), 4),
                'val_perplexity': round(val_perplexity, 2),
                'time': round(elapsed, 2)
            })
            
    total_time = time.time() - start_time
    opt_mem_after = get_optimizer_memory(optimizer)
    
    results['total_time'] = round(total_time, 2)
    results['optimizer_memory_mb'] = round(opt_mem_after, 2)
    results['final_val_loss'] = results['iters'][-1]['val_loss']
    results['final_val_perplexity'] = results['iters'][-1]['val_perplexity']
    
    print(f"\nFinal Results:")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Final Val Perplexity: {results['final_val_perplexity']:.2f}")
    print(f"  Optimizer state memory: {opt_mem_after:.2f} MB")
    
    return results

# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SMO variants vs AdamW on MiniGPT (smoke test)")
    parser.add_argument('--max_iters', type=int, default=200, help='Maximum training iterations')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducibility')
    args = parser.parse_args()

    device = 'cpu'
    max_iters = args.max_iters
    seed = args.seed
    
    print("="*60)
    print("LLM Benchmark: SMO variants vs AdamW on Mini-GPT")
    print("="*60)
    print(f"Device: {device}")
    print(f"Max Iters: {max_iters}")
    print(f"Seed: {seed}")
    
    # 1. Baseline AdamW
    results_adam = run_experiment(
        "Standard AdamW",
        lambda params: torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2),
        max_iters=max_iters,
        device=device,
        seed=seed
    )
    
    # 2. SMO (Spatial)
    results_smo = run_experiment(
        "SMO (k_ratio=0.5)",
        lambda params: SMO(params, lr=1e-3, weight_decay=1e-2, k_ratio=0.5),
        max_iters=max_iters,
        device=device,
        seed=seed
    )
    
    # 3. SMO-8bit (Extreme Compression)
    results_8bit = run_experiment(
        "SMO-8bit (k_ratio=0.5)",
        lambda params: SMO8bit(params, lr=1e-3, weight_decay=1e-2, k_ratio=0.5),
        max_iters=max_iters,
        device=device,
        seed=seed
    )
    
    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    all_results = [results_adam, results_smo, results_8bit]
        
    for r in all_results:
        mem = r['optimizer_memory_mb']
        ppl = r['final_val_perplexity']
        print(f"\n{r['optimizer']}:")
        print(f"  Final Val Perplexity: {ppl:.2f}")
        print(f"  Optimizer Memory:     {mem:.2f} MB")
    
    # Save results
    out_dict = {
        'adamw': results_adam,
        'smo_05': results_smo,
        'smo_8bit_05': results_8bit,
    }
    runs = [
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="AdamW",
            script_name="benchmarks/suites/training/benchmark_minillm.py",
            hardware=device.upper(),
            backend="CPU",
            dataset="SyntheticTokens",
            model="MiniGPT",
            batch_size=16,
            precision="fp32",
            steps=max_iters,
            seed=seed,
            metrics={
                "final_val_loss": results_adam['final_val_loss'],
                "final_val_perplexity": results_adam['final_val_perplexity'],
                "optimizer_state_mb": results_adam['optimizer_memory_mb'],
                "total_time_s": results_adam['total_time'],
            },
            extra={"raw_result": results_adam},
        ),
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="SMO-Spatial",
            script_name="benchmarks/suites/training/benchmark_minillm.py",
            hardware=device.upper(),
            backend="CPU",
            dataset="SyntheticTokens",
            model="MiniGPT",
            batch_size=16,
            precision="fp32",
            steps=max_iters,
            seed=seed,
            metrics={
                "final_val_loss": results_smo['final_val_loss'],
                "final_val_perplexity": results_smo['final_val_perplexity'],
                "optimizer_state_mb": results_smo['optimizer_memory_mb'],
                "total_time_s": results_smo['total_time'],
            },
            extra={"raw_result": results_smo, "k_ratio": 0.5},
        ),
        make_run_record(
            benchmark_family="end_to_end_training",
            variant="SMO-Spatial-8bit",
            script_name="benchmarks/suites/training/benchmark_minillm.py",
            hardware=device.upper(),
            backend="CPU",
            dataset="SyntheticTokens",
            model="MiniGPT",
            batch_size=16,
            precision="fp32",
            steps=max_iters,
            seed=seed,
            metrics={
                "final_val_loss": results_8bit['final_val_loss'],
                "final_val_perplexity": results_8bit['final_val_perplexity'],
                "optimizer_state_mb": results_8bit['optimizer_memory_mb'],
                "total_time_s": results_8bit['total_time'],
            },
            extra={"raw_result": results_8bit, "k_ratio": 0.5},
        ),
    ]
    results_path, _ = write_benchmark_bundle(
        aggregate_filename="benchmark_minillm_results.json",
        suite_name="minillm_training_comparison",
        benchmark_family="end_to_end_training",
        summary=out_dict,
        runs=runs,
    )
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()