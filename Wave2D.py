import numpy as np
import sympy as sp
from scipy import sparse

x, y, t = sp.symbols("x,y,t")

class Wave2D:
    def create_mesh(
        self, N: int, sparse: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        self.N = N
        self.h = 1.0 / N
        xi = np.linspace(0, 1, N + 1)
        self.xij, self.yij = np.meshgrid(xi, xi, indexing="ij", sparse=sparse)
        return self.xij, self.yij

    def D2(self, N: int) -> sparse.lil_matrix:

        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        kx = self.mx * sp.pi
        ky = self.my * sp.pi
        return self.c * sp.sqrt(kx**2 + ky**2)

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N: int, mx: int, my: int) -> np.ndarray:
        self.mx, self.my = mx, my
        self.Unp1, self.Un, self.Unm1 = np.zeros((3, N + 1, N + 1))
        self.Unm1[:] = sp.lambdify((x, y, t), self.ue(mx, my), "numpy")(
            self.xij, self.yij, 0
        )
        D = self.D2(N) / self.h**2
        self.Un[:] = self.Unm1 + 0.5 * (self.c * self.dt) ** 2 * (
            D @ self.Unm1 + self.Unm1 @ D.T
        )
        self.apply_bcs(self.Un)
        return self.Un

    @property
    def dt(self) -> float:
        return self.cfl * self.h / self.c

    def l2_error(self, u: np.ndarray, t0: float) -> float:
        uej = sp.lambdify((x, y, t), self.ue(self.mx, self.my), "numpy")(
            self.xij, self.yij, t0
        )
        return float(np.sqrt(self.h**2 * np.sum((u - uej) ** 2)))

    def apply_bcs(self, u: np.ndarray):
        u[0] = 0
        u[-1] = 0
        u[:, 0] = 0
        u[:, -1] = 0

    def __call__(
        self,
        N: int,
        Nt: int,
        cfl: float = 0.5,
        c: float = 1.0,
        mx: int = 3,
        my: int = 3,
        store_data: int = -1,
    ):
        self.cfl = cfl
        self.c = c
        self.create_mesh(N)
        self.initialize(N, mx, my)
        D = self.D2(N) / self.h**2
        dt = self.dt

        data = {}
        err = []
        if store_data > 0:
            data[0] = self.Unm1.copy()
            if store_data == 1:
                data[1] = self.Un.copy()
        else:
            err.append(self.l2_error(self.Unm1, 0))
            err.append(self.l2_error(self.Un, dt))

        for n in range(1, Nt):
            self.Unp1[:] = (
                2 * self.Un
                - self.Unm1
                + (c * dt) ** 2 * (D @ self.Un + self.Un @ D.T)
            )
            self.apply_bcs(self.Unp1)
            self.Unm1, self.Un, self.Unp1 = self.Un, self.Unp1, self.Unm1
            if store_data > 0 and (n + 1) % store_data == 0:
                data[n + 1] = self.Un.copy()
            elif store_data == -1:
                err.append(self.l2_error(self.Un, (n + 1) * dt))

        if store_data > 0:
            return data
        return self.h, np.array(err)

    def convergence_rates(
        self, m: int = 4, cfl: float = 0.1, Nt: int = 10, mx: int = 3, my: int = 3
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m, 1)
        ]
        return np.array(r), np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):
    def D2(self, N: int) -> sparse.lil_matrix:
        D = sparse.diags([1.0, -2.0, 1.0], [-1, 0, 1], (N + 1, N + 1), format="lil")
        D[0, :2] = -2, 2
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx: int, my: int) -> sp.Expr:
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self, u: np.ndarray):
        pass


def make_movie(
    filename: str = "report/neumannwave.gif",
    N: int = 40,
    Nt: int = 80,
    store_data: int = 2,
    fps: int = 15,
):

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    sol = Wave2D_Neumann()
    data = sol(N, Nt, cfl=1 / np.sqrt(2), mx=2, my=2, store_data=store_data)
    xij, yij = sol.xij, sol.yij
    dt = sol.dt

    fig = plt.figure(figsize=(5, 4), dpi=70)
    ax = fig.add_subplot(projection="3d")
    frames = []
    for n, U in data.items():
        surf = ax.plot_surface(
            xij, yij, U, cmap="viridis", vmin=-1, vmax=1, linewidth=0, antialiased=False
        )
        title = ax.text2D(0.5, 0.95, f"t = {n * dt:.3f}", transform=ax.transAxes,
                          ha="center")
        frames.append([surf, title])
    ax.set_zlim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ani = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True,
                                    repeat_delay=500)
    ani.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(m=5, mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2, r


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=3, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    cfl = 1 / np.sqrt(2)
    for sol in (Wave2D(), Wave2D_Neumann()):
        for m in (1, 2, 3):
            _, err = sol(N=20, Nt=50, cfl=cfl, mx=m, my=m, store_data=-1)
            assert np.max(err) < 1e-12, (type(sol).__name__, m, np.max(err))


if __name__ == "__main__":
    test_convergence_wave2d()
    test_convergence_wave2d_neumann()
    test_exact_wave2d()
    print("All tests passed!")
