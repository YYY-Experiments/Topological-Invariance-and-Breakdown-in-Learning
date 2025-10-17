import numpy as np
import torch as tc
import math

def initialize_S2(n: int, d: int, radius = 1.0) -> tc.Tensor:

    A = tc.randn(d, 3)
    Q, _ = tc.linalg.qr(A, mode="reduced") 

    X = tc.randn(n, 3)
    X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-12)

    Xd = (X @ Q.t()) * radius
    return Xd
