import numpy as np


def tv_norm_1d(x):
    return np.sum(np.abs(np.diff(x)))


def prox_group_lasso(x, lam):
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    scale = max(0.0, 1.0 - lam / norm)
    return scale * x


def prox_tv1d_condat(y, lam):
    """
    Proximal operator of 1D total variation using Condat's algorithm.
    Reference: Condat, "A Direct Algorithm for 1-D Total Variation Denoising", 2013.
    """
    n = len(y)
    if n == 0 or lam <= 0:
        return y.copy()

    x = np.empty(n, dtype=float)
    k = k0 = 0
    vmin = y[0] - lam
    vmax = y[0] + lam
    umin = lam
    umax = -lam

    while True:
        if k == n - 1:
            if umin < 0:
                for i in range(k0, k + 1):
                    x[i] = vmin
            elif umax > 0:
                for i in range(k0, k + 1):
                    x[i] = vmax
            else:
                v = y[k] + umin / (k - k0 + 1)
                for i in range(k0, k + 1):
                    x[i] = v
            break

        k += 1
        val = y[k]
        umin += val - vmin
        umax += val - vmax

        if umin > lam:
            vmin += (umin - lam) / (k - k0 + 1)
            umin = lam
        if umax < -lam:
            vmax += (umax + lam) / (k - k0 + 1)
            umax = -lam

        if umin >= lam:
            for i in range(k0, k + 1):
                x[i] = vmin
            k0 = k + 1
            if k0 >= n:
                break
            k = k0
            vmin = y[k] - lam
            vmax = y[k] + lam
            umin = lam
            umax = -lam
            continue
        if umax <= -lam:
            for i in range(k0, k + 1):
                x[i] = vmax
            k0 = k + 1
            if k0 >= n:
                break
            k = k0
            vmin = y[k] - lam
            vmax = y[k] + lam
            umin = lam
            umax = -lam
            continue

    return x


def prox_tv_group_lasso(v, lam_tv, lam_gl, iters=25):
    """
    Prox for lam_tv * TV(x) + lam_gl * ||x||_2 using Dykstra iterations.
    """
    if lam_tv <= 0 and lam_gl <= 0:
        return v.copy()

    x = v.copy()
    p = np.zeros_like(v)
    q = np.zeros_like(v)
    for _ in range(iters):
        y = prox_tv1d_condat(x + p, lam_tv)
        p = x + p - y
        x = prox_group_lasso(y + q, lam_gl)
        q = y + q - x
    return x
