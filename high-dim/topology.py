import torch as tc
import numpy as np
import gudhi
from scipy.spatial.distance import pdist

def betti_numbers(points: tc.Tensor):

    n,d = points.size()
    X = points.view(n,d).detach().cpu().numpy()

    diam = pdist(X).max()
    eps = diam * 0.25

    rips_complex = gudhi.RipsComplex(points=X, max_edge_length=eps)
    st = rips_complex.create_simplex_tree(max_dimension=3)
    st.persistence()

    return st.betti_numbers()[:3]
