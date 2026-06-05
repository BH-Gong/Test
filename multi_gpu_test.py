import torch
import time

def bench(dtype, N=4096, iters=50, device="cuda"):
    if dtype == torch.float64:
        N = 2048

    if dtype in [torch.int8, torch.int32]:
        a = torch.randint(-5, 5, (N, N), device=device, dtype=dtype)
        b = torch.randint(-5, 5, (N, N), device=device, dtype=dtype)
    else:
        a = torch.randn(N, N, device=device, dtype=dtype)
        b = torch.randn(N, N, device=device, dtype=dtype)

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

    bench(torch.int32)
    bench(torch.int8)

if __name__ == "__main__":
    main()