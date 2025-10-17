import numpy as np
import torch as tc
import math

def twist_map(x, y, alpha=1.2, sigma=0.8):
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    theta2 = theta + alpha * np.exp(-(r/sigma)**2)
    xr = r * np.cos(theta2)
    yr = r * np.sin(theta2)
    return xr, yr

def alt_initialize_2d(H, radius_top=1.0, radius_bot=1.0, gap=2.4):
    H_top = H // 2
    H_bot = H - H_top

    th_top = tc.linspace(0, 2*math.pi, H_top+1)[:-1]
    th_bot = tc.linspace(0, 2*math.pi, H_bot+1)[:-1]

    cx_top, cy_top = 0.0, +gap/2.0
    cx_bot, cy_bot = 0.0, -gap/2.0

    x_top = cx_top + radius_top * tc.cos(th_top)
    y_top = cy_top + radius_top * tc.sin(th_top)
    x_bot = cx_bot + radius_bot * tc.cos(th_bot)
    y_bot = cy_bot + radius_bot * tc.sin(th_bot)

    xs = tc.cat([x_top, x_bot], dim=0)
    ys = tc.cat([y_top, y_bot], dim=0)

    for i in range(len(xs)):
        x, y = xs[i].item(), ys[i].item()

        # uncoment to use a twisted initialization shape
        # x, y = twist_map(x, y, alpha=1.2, sigma=0.8)

        xs[i] = x
        ys[i] = y

    return xs, ys


def initialize_2d(n):

    t = np.random.uniform(0, 2*np.pi, n)
    xs = []
    ys = []
    for i in range(n):
        xs.append( float(np.sin(t[i])) )
        ys.append( float(np.sin(2*t[i])) )

    return xs, ys

def initialize_3d(n):
    def euler_matrix_xyz(rx, ry, rz):
        cx, sx = tc.cos(tc.tensor(rx)), tc.sin(tc.tensor(rx))
        cy, sy = tc.cos(tc.tensor(ry)), tc.sin(tc.tensor(ry))
        cz, sz = tc.cos(tc.tensor(rz)), tc.sin(tc.tensor(rz))

        Rx = tc.tensor([[1, 0, 0],
                        [0, cx, -sx],
                        [0, sx,  cx]], dtype=tc.float32)
        Ry = tc.tensor([[ cy, 0, sy],
                        [  0, 1,  0],
                        [-sy, 0, cy]], dtype=tc.float32)
        Rz = tc.tensor([[cz, -sz, 0],
                        [sz,  cz, 0],
                        [ 0,   0, 1]], dtype=tc.float32)
        return (Rz @ Ry @ Rx)


    def soft_min(a: tc.Tensor, b: tc.Tensor, k: float) -> tc.Tensor:
        return -tc.logsumexp(tc.stack([-k * a, -k * b], dim=0), dim=0) / k

    def torus_F(x, y, z, cx, R, r):
        rho = tc.sqrt((x - cx) ** 2 + y ** 2)
        return (rho - R) ** 2 + z ** 2 - r ** 2

    def double_torus_F(x, y, z, R, r, d, k, scale_z):
        z_unscaled = z / scale_z

        f1 = torus_F(x, y, z_unscaled, +d, R, r)
        f2 = torus_F(x, y, z_unscaled, -d, R, r)
        return soft_min(f1, f2, k)

    R = 0.5 
    r = 0.2 
    d = 0.8 
    k = 5.0 
    scale_z = 1
    batch_size = 10000
    device = "cuda"
    angles = (0.6 , 0.3, 1.1)
    Rot = euler_matrix_xyz(*angles).to(device)

    x_min = -d - (R + r)
    x_max =  d + (R + r)
    y_min = -(R + r)
    y_max = +(R + r)
    z_min = -r * 1.05 * scale_z
    z_max = +r * 1.05 * scale_z

    pts = []
    needed = n
    while needed > 0:
        m = max(batch_size, needed) 
        xs = tc.empty(m, device=device).uniform_(x_min, x_max)
        ys = tc.empty(m, device=device).uniform_(y_min, y_max)
        zs = tc.empty(m, device=device).uniform_(z_min, z_max)

        F = double_torus_F(xs, ys, zs, R, r, d, k, scale_z)
        inside = F <= 0.0
        if inside.any():
            sel = tc.stack([xs[inside], ys[inside], zs[inside]], dim=1)
            if sel.shape[0] > needed:
                sel = sel[:needed]
            pts.append(sel)
            needed -= sel.shape[0]

    P = tc.cat(pts, dim=0) 
    P = P @ Rot.T 

    return P[:,0], P[:,1], P[:,2]
