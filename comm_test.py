import torch
import torch.distributed as dist
import os
import time

def run():
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world = dist.get_world_size()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # tensor size (조절 가능)
    tensor = torch.ones(1024 * 1024, device=device)

    # warmup
    for _ in range(10):
        dist.all_reduce(tensor)

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(50):
        dist.all_reduce(tensor)

    torch.cuda.synchronize()

    elapsed = time.time() - start

    bytes_per_iter = tensor.numel() * 4 * 2  # float32 + allreduce (rough)
    bandwidth = (bytes_per_iter * 50) / elapsed / 1e9

    if rank == 0:
        print(f"AllReduce Bandwidth: {bandwidth:.2f} GB/s")

    dist.destroy_process_group()

if __name__ == "__main__":
    run()