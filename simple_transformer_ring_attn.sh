export OMP_NUM_THREADS=4
torchrun --nproc-per-node 1 simple_transformer_ring_attn.py
torchrun --nproc-per-node 2 simple_transformer_ring_attn.py
torchrun --nproc-per-node 4 simple_transformer_ring_attn.py
torchrun --nproc-per-node 8 simple_transformer_ring_attn.py


