"""
One-sided vs two-sided folding for the 1D advection--diffusion operator

    A u = -nu u'' - u',   D(A) = H^2(0,1) n H^1_0(0,1),   nu = 0.1,
    A* u = -nu u'' + u',  D(A*) = D(A).

For each z we compute, in a conforming cubic Hermite space V_h c D(A) = D(A*),

    mu1(z) = min_{u_h} ||(A  - z   ) u_h||^2 / ||u_h||^2     ("A*A", eq. aux1)
    mu2(z) = min_{u_h} ||(A* - zbar) u_h||^2 / ||u_h||^2     ("AA*", eq. aux2)

and Phi_h = sqrt(min(mu1, mu2)).  The Rayleigh quotients are evaluated as the
smallest singular value of the quadrature-weighted "collocation" matrix
W^{1/2} [(A - z) phi_j](x_q) L^{-T}, M = L L^T, which equals sqrt(mu) exactly
(Gauss quadrature is exact for these integrands) but avoids squaring the
condition number.  The two problems are solved completely independently.

A Chebyshev collocation computation of ||(A-z)^{-1}||^{-1} is the reference.
"""
import sys
import numpy as np
from scipy.linalg import cholesky, solve_triangular, svdvals

NU = 0.1


# ----------------------------------------------------------------------------
# cubic Hermite, conforming in H^2 n H^1_0
# ----------------------------------------------------------------------------
def hermite_tables(nodes, sign, nq=6):
    """Return (F0, FA): quadrature-weighted values of phi_j and of
    (-nu phi_j'' - sign*phi_j') at all Gauss points.  sign=+1 -> A, -1 -> A*."""
    nel = len(nodes) - 1
    xi, w = np.polynomial.legendre.leggauss(nq)
    t = 0.5 * (xi + 1.0)                       # reference points in [0,1]
    w = 0.5 * w
    # reference shape functions (value/deriv at left, value/deriv at right)
    N = np.array([1 - 3*t**2 + 2*t**3, t - 2*t**2 + t**3,
                  3*t**2 - 2*t**3, -t**2 + t**3])
    dN = np.array([-6*t + 6*t**2, 1 - 4*t + 3*t**2,
                   6*t - 6*t**2, -2*t + 3*t**2])
    ddN = np.array([-6 + 12*t, -4 + 6*t, 6 - 12*t, -2 + 6*t])

    ndof_full = 2 * (nel + 1)
    F0 = np.zeros((nel * nq, ndof_full))
    FA = np.zeros((nel * nq, ndof_full))
    for e in range(nel):
        h = nodes[e + 1] - nodes[e]
        scale = np.array([1.0, h, 1.0, h])[:, None]   # derivative dofs scale with h
        phi = N * scale
        dphi = dN * scale / h
        ddphi = ddN * scale / h**2
        sw = np.sqrt(w * h)
        dofs = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
        rows = slice(e * nq, (e + 1) * nq)
        F0[rows, dofs] = (phi * sw).T
        FA[rows, dofs] = ((-NU * ddphi - sign * dphi) * sw).T
    keep = np.ones(ndof_full, bool)
    keep[0] = keep[-2] = False                  # u(0) = u(1) = 0
    return F0[:, keep], FA[:, keep]


class Folded:
    """sigma_min of (B - s) restricted to V_h, B = A (sign=+1) or A* (sign=-1)."""

    def __init__(self, nodes, sign):
        F0, FA = hermite_tables(nodes, sign)
        M = F0.T @ F0
        L = cholesky(M, lower=True)
        # M-orthonormalise the basis: columns of F0 L^{-T} are orthonormal
        self.F0 = solve_triangular(L, F0.T, lower=True).T
        self.FA = solve_triangular(L, FA.T, lower=True).T

    def sigma(self, s):
        return svdvals(self.FA - s * self.F0)[-1]


def sweep(nodes, X, Y):
    fa, fs = Folded(nodes, +1), Folded(nodes, -1)
    s1 = np.empty(X.shape)
    s2 = np.empty(X.shape)
    for idx in np.ndindex(X.shape):
        z = X[idx] + 1j * Y[idx]
        s1[idx] = fa.sigma(z)                   # sqrt(mu1): (A - z)
        s2[idx] = fs.sigma(np.conj(z))          # sqrt(mu2): (A* - zbar)
    return s1, s2


# ----------------------------------------------------------------------------
# Chebyshev reference for gamma(z, A) = 1 / ||(A - z)^{-1}||
# ----------------------------------------------------------------------------
def cheb(n):
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    Xm = np.tile(x, (n + 1, 1)).T
    dX = Xm - Xm.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    return D - np.diag(D.sum(axis=1)), x


def clencurt(n):
    theta = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    ii = np.arange(1, n)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
        v -= np.cos(n * theta[ii]) / (n**2 - 1)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[ii]) / (4 * k**2 - 1)
    w[ii] = 2 * v / n
    return w


def reference(X, Y, n=96):
    D, x = cheb(n)
    D = 2.0 * D                                  # [-1,1] -> [0,1]
    A = (-NU * D @ D - D)[1:-1, 1:-1]
    w = 0.5 * clencurt(n)[1:-1]
    sw = np.sqrt(w)
    Aw = (sw[:, None] * A) / sw[None, :]         # L^2-weighted similarity
    I = np.eye(n - 1)
    out = np.empty(X.shape)
    for idx in np.ndindex(X.shape):
        out[idx] = svdvals(Aw - (X[idx] + 1j * Y[idx]) * I)[-1]
    return out


if __name__ == "__main__":
    out = sys.argv[1]
    nx, ny = 121, 161
    X, Y = np.meshgrid(np.linspace(0, 15, nx), np.linspace(-10, 10, ny))

    nel = 16                                     # the thesis' initial mesh
    uniform = np.linspace(0, 1, nel + 1)
    graded = np.linspace(0, 1, nel + 1) ** 3     # clustered at x=0, coarse at x=1
    graded_x1 = (1.0 - graded)[::-1]             # its mirror image: clustered at x=1

    res = {"X": X, "Y": Y, "uniform": uniform, "graded": graded,
           "graded_x1": graded_x1}
    res["ref"] = reference(X, Y)
    res["u1"], res["u2"] = sweep(uniform, X, Y)
    res["g1"], res["g2"] = sweep(graded, X, Y)
    res["r1"], res["r2"] = sweep(graded_x1, X, Y)
    np.savez(out, **res)
    print("saved", out)
