import numpy as np
import sympy as sp
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from poisson import Poisson

x, y = sp.symbols("x,y")

class Poisson2D:
    def __init__(self, L: float):
        self.p = Poisson(L)

    def create_mesh(self, N: int) -> tuple[np.ndarray, np.ndarray]:

        xi = self.p.create_mesh(N)
        xij, yij = np.meshgrid(xi, xi, indexing="ij", sparse=True)
        return xij, yij

    def laplace(self, N: int) -> sparse.lil_matrix:
        h = self.p.L / N
        D2 = self.p.D2(N, h)
        I = sparse.eye(N + 1)
        return (sparse.kron(D2, I) + sparse.kron(I, D2)).tolil()

    def assemble(
        self, N: int, f: sp.Expr, ue: sp.Expr
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        xij, yij = self.create_mesh(N)
        A = self.laplace(N)
        bnds = self.get_boundary_indices(N)
        A[bnds] = 0
        A[bnds, bnds] = 1
        b = self.meshfunction(f, xij, yij).ravel()
        b[bnds] = self.meshfunction(ue, xij, yij).ravel()[bnds]
        return A.tocsr(), b

    def meshfunction(self, u: sp.Expr, xij: np.ndarray, yij: np.ndarray) -> np.ndarray:
        shape = np.broadcast_shapes(xij.shape, yij.shape)
        values = sp.lambdify((x, y), u, "numpy")(xij, yij)
        return np.array(np.broadcast_to(values, shape), dtype=float)

    def get_boundary_indices(self, N: int) -> np.ndarray:
        B = np.ones((N + 1, N + 1), dtype=bool)
        B[1:-1, 1:-1] = False
        return np.flatnonzero(B)

    def l2_error(self, u: np.ndarray, ue: sp.Expr) -> float:
        N = u.shape[0] - 1
        h = self.p.L / N
        xij, yij = self.create_mesh(N)
        uej = self.meshfunction(ue, xij, yij)
        return float(np.sqrt(h**2 * np.sum((u - uej) ** 2)))

    def __call__(self, N: int, ue: sp.Expr) -> np.ndarray:
        A, b = self.assemble(N, sp.diff(ue, x, 2) + sp.diff(ue, y, 2), ue)
        return sparse_linalg.spsolve(A, b.ravel()).reshape((N + 1, N + 1))

    def convergence_rates(self, ue: sp.Expr, m: int = 6):
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            u = self(N0, ue)
            E.append(self.l2_error(u, ue))
            h.append(self.p.L / N0)
            N0 *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)

    @staticmethod
    def _lagrange_weights(xj: np.ndarray, xp: float) -> np.ndarray:
        n = len(xj)
        w = np.ones(n)
        for k in range(n):
            for m in range(n):
                if m != k:
                    w[k] *= (xp - xj[m]) / (xj[k] - xj[m])
        return w

    def eval(self, U: np.ndarray, x: float, y: float, order: int = 3) -> float:
        N = U.shape[0] - 1
        L = self.p.L
        if not (0 <= x <= L and 0 <= y <= L):
            raise ValueError("Point (x, y) is outside the domain")
        h = L / N
        npts = order + 1
        xi = self.p.create_mesh(N)

        def first_index(p: float) -> int:
            i = min(int(p // h), N - 1)
            i0 = i - (npts - 1) // 2
            return int(np.clip(i0, 0, N + 1 - npts))

        i0 = first_index(float(x))
        j0 = first_index(float(y))
        lx = self._lagrange_weights(xi[i0 : i0 + npts], float(x))
        ly = self._lagrange_weights(xi[j0 : j0 + npts], float(y))
        return float(lx @ U[i0 : i0 + npts, j0 : j0 + npts] @ ly)

def test_convergence_poisson2d():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    r, _, _ = sol.convergence_rates(ue)
    assert abs(r[-1] - 2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4 * sp.pi * x) * sp.sin(2 * sp.pi * y))
    sol = Poisson2D(1)
    N = 100
    U = sol(N, ue)
    h = sol.p.L / N
    assert abs(sol.eval(U, 0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(U, h / 2, 1 - h / 2) - ue.subs({x: h, y: 1 - h / 2}).n()) < 1e-3

if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()
    print("All tests passed!")
