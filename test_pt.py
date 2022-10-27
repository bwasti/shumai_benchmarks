import torch
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pw(size, iters, warmup):
    a = torch.randn(size).to(device)

    def op(K):
        b = a
        for i in range(K):
            b = b + a
        o = b.sum().item()

    op(warmup)
    t0 = time.time()
    op(iters)
    t1 = time.time()

    bs = size * 4 * iters + size * 4
    gbs = bs / (1e9 * (t1 - t0))
    kitersec = iters / ((t1 - t0) * 1e3)
    print(f"{kitersec:.3f}K iter/s", f"{gbs:.3f} GB/s")


def mm(size, iters, warmup):
    N = size
    a = torch.randn(N, N).to(device)
    b = torch.eye(N).to(device)

    def op(K):
        c = a
        for i in range(K):
            c = c @ b
        o = c.sum().item()

    op(warmup)
    t0 = time.time()
    op(iters)
    t1 = time.time()

    flops = N**3 * 2 * iters + N * N
    gflops = flops / (1e9 * (t1 - t0))
    print(f"{gflops:.3f} GFlop/s")


def mm_pw(size, iters, warmup):
    N = size
    a = torch.randn(64, N).to(device)
    b = torch.eye(N).to(device)
    c = torch.randn(1, N).to(device)

    def op(K):
        d = a
        for i in range(K):
            d = d @ b
            for j in range(5):
                d = d + c
        o = d.sum().item()

    op(warmup)
    t0 = time.time()
    op(iters)
    t1 = time.time()

    kitersec = iters / ((t1 - t0) * 1e3)
    print(f"{kitersec:.3f}K iter/s")

def mha(size, iters, warmup):
  N = size
  mha_model = torch.nn.MultiheadAttention(N, 8).eval().to(device)
  qk = torch.rand((32, 32, N)).to(device)
  v = torch.rand((32, 32, N)).to(device)
  def op(K):
     for i in range(K):
       out = mha_model(qk, qk, v)[0]
     o = out.sum().item()

  op(warmup)
  t0 = time.time()
  op(iters)
  t1 = time.time()

  itersec = iters / (t1 - t0)
  print(f"{itersec:.3f} iter/s")

def err(n):
    print(f"usage: python {n} {pw,mm} size iters warmup")
    exit(1)


if len(sys.argv) < 5:
    err(sys.argv[0])
var = sys.argv[1]
if not var in ["mm", "pw", "mm_pw", "mha"]:
    err(sys.argv[0])
size = int(sys.argv[2])
iters = int(sys.argv[3])
warmup = int(sys.argv[4])

if var == "pw":
    pw(size, iters, warmup)
elif var == "mm_pw":
    mm_pw(size, iters, warmup)
elif var == "mm":
    mm(size, iters, warmup)
elif var == "mha":
    mha(size, iters, warmup)
