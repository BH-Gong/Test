import torch
import time

def bench(device, dtype, N=4096, iters=50):
    if dtype == torch.float64:
        N = 2048

    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)

    # warmup
    for _ in range(10):
        torch.matmul(a, b)

    torch.cuda.synchronize(device)

    start = time.time()

    for _ in range(iters):
        torch.matmul(a, b)

    torch.cuda.synchronize(device)

    elapsed = time.time() - start

    flops = iters * 2 * (N ** 3)
    tflops = flops / elapsed / 1e12

    return tflops


def main():
    n_gpus = torch.cuda.device_count()
    print(f"Detected GPUs: {n_gpus}")

    dtypes = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64
    ]

    for gpu_id in range(n_gpus):
        device = f"cuda:{gpu_id}"
        print(f"\n===== GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)} =====")

        for dtype in dtypes:
            try:
                tflops = bench(device, dtype)
                print(f"{dtype}: {tflops:.2f} TFLOPS")
            except Exception as e:
                print(f"{dtype}: ERROR -> {e}")


if __name__ == "__main__":
    main()