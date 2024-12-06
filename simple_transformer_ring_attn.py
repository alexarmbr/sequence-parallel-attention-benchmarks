import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.distributed.tensor.experimental._attention import _templated_ring_attention
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed._functional_collectives as ft_c
import os
import time

# set environment variables

WORLD_SIZE = int(os.environ["WORLD_SIZE"])
SEQ_LEN = 33792
HEADDIM = 128
SEED = 42
ITERATIONS = 50
N_LAYERS = 92
ATTN_FN = torch.ops.aten._scaled_dot_product_flash_attention
MESH = DeviceMesh(device_type="cuda", mesh=list(range(WORLD_SIZE)))

# uses pytorch 2.5 experimental ring attention
# ring attention alone performs poorly

class Model(torch.nn.Module):
    def __init__(self, n_layers, attn_fn, embed_dim):
        super().__init__()
        self.layers = torch.nn.ModuleList([AttentionBlock(embed_dim, attn_fn) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, attn_fn, seed = 42):
        # make results deterministic across different instantiations of this class
        torch.manual_seed(seed)
        super().__init__()
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.o_proj = torch.nn.Linear(embed_dim, embed_dim, dtype=torch.bfloat16)
        self.attn_fn = attn_fn

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out, *_ = self.attn_fn(q, k, v)
        return self.o_proj(out)


def setup_distributed(world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=local_rank, world_size=world_size, device_id=device)
    
    return dist.is_initialized()

def get_local_rank():
    return int(os.environ["LOCAL_RANK"])

def get_device():
    return torch.device(f"cuda:{get_local_rank()}")

def ring_attn(query, key, value, dropout_p=0.0, is_causal=False):
    out, *_ = _templated_ring_attention(
        MESH,
        ATTN_FN,
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal
    )
    out = out.unsqueeze(0)
    return out


def benchmark_ring_attn(X):
    local_rank = get_local_rank()
    device = get_device()
    X_local = X.chunk(WORLD_SIZE, dim=2)[local_rank]
    dist_attn_layers = Model(N_LAYERS, embed_dim=HEADDIM, attn_fn=ring_attn).to(device)
    times = []
    with torch.inference_mode():
        for i in range(ITERATIONS):
            t0 = time.time()
            dist_O = dist_attn_layers(X_local.clone())
            dist_O = ft_c.all_gather_tensor(dist_O, gather_dim=2, group=MESH)
            t1 = time.time()
            times.append(t1 - t0)
        
        times = times[len(times) // 2:]
        avg_time = sum(times) / len(times)
    return avg_time, dist_O

def benchmark_regular_attn(X):
    device = get_device()
    attn_layers = Model(N_LAYERS, embed_dim=HEADDIM, attn_fn = ATTN_FN).to(device)

    with torch.inference_mode():
        times = []
        for i in range(ITERATIONS):
            t0 = time.time()
            O = attn_layers(X.clone())
            t1 = time.time()
            times.append(t1 - t0)
        
        times = times[len(times) // 2:]
        avg_time = sum(times) / len(times)
    return avg_time, O

def main():
    
    # set seed an generate a single tensor that will be input to the model
    setup_distributed(WORLD_SIZE)
    device = get_device()
    X = torch.randn(1, 1, SEQ_LEN, HEADDIM, device=device, dtype=torch.bfloat16)

    ring_time, ring_O = benchmark_ring_attn(X)
    if get_local_rank() == 0:
        regular_time, regular_O = benchmark_regular_attn(X)
        print(f"# gpus: {WORLD_SIZE}, ring time: {ring_time}, regular time: {regular_time}")
        assert torch.allclose(ring_O, regular_O, rtol=1e-3, atol=1e-3)
    
    dist.destroy_process_group()


        


if __name__ == "__main__":
    main()
    