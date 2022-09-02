#!/usr/bin/env python3
# Single precision (float32) - General matrix multiplier

import time
import numpy as np

N = 4096
if __name__ == "__main__":
    #N^2
    a = np.random.randn(N, N).astype(np.float32)
    #N^2
    b = np.random.randn(N, N).astype(np.float32)

    # N^2 output cells with 2N compute each (+&*) - FLOP (Floating point operation)
    flop = N*N*2*N
    print(f"{flop / 1e9:.2f} GFLOP")

    st = time.monotonic()
    c = a @ b
    et = time.monotonic()
    s = et-st
    # print(c)

    print(f"{flop/s * 1e-9:.2f} GFLOP/S", f"-- {s:.2f} Secs",)
