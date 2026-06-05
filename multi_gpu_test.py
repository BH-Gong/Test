import torch
import time

def bench(dtype, N=4096, iters=2000):
    device = "cuda"

    # FP64는 너무 느리니까 크기 줄임
    if dtype == torch.float64:
        N = 2048

    a = torch.randn(N, N, device=device, dtype=dtype)
    b = torch.randn(N, N, device=device, dtype=dtype)

    # warmup
    for _ in range(10):
        torch.matmul(a, b)

    torch.cuda.synchronize()

    start = time.time()

    for _ in range(iters):
        torch.matmul(a, b)

    torch.cuda.synchronize()

    elapsed = time.time() - start

    flops = iters * 2 * (N ** 3)
    tflops = flops / elapsed / 1e12

    print(f"{dtype} | N={N} | {tflops:.2f} TFLOPS")

def main():
    bench(torch.float16)
    bench(torch.bfloat16)
    bench(torch.float32)
    bench(torch.float64)

if __name__ == "__main__":
    main()