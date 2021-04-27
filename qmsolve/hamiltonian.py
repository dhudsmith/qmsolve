import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator, expm_multiply
from typing import Callable, List, Tuple, Union
from abc import ABC, abstractmethod


class DiscreteSpace:
    def __init__(self,
                 dim: int,
                 support: Union[Tuple[float, float], List[Tuple[float, float]]],
                 grid: Union[int, List[int]],
                 dtype=np.float64):

        if isinstance(support, List):
            assert len(support) == dim, \
                "Support must be a 2-tuple of floats or a List of 2-tuples with length equal to spacial_dim"
        if isinstance(grid, List):
            assert len(grid) == dim, \
                "Grid must be an integer or a List of integers with length equal to dim"

        self.dim = dim
        self.dtype = dtype
        self.support = support if isinstance(support, List) else [support] * self.dim
        self.grid = grid if isinstance(grid, List) else [grid] * self.dim
        self.grid_points, self.steps = self._compute_grid()

    def _compute_grid(self):
        grid_points = [np.linspace(self.support[i][0], self.support[i][1], self.grid[i], dtype=self.dtype)
                       for i in range(self.dim)]
        steps = [ar[1] - ar[0] for ar in grid_points]

        return grid_points, steps

    def vec_to_mesh(self, vec: np.ndarray) -> np.ndarray:
        return vec.reshape(-1, *self.grid)

    def mesh_to_vec(self, mesh: np.ndarray) -> np.ndarray:
        return mesh.reshape(-1, np.prod(self.grid))


class Hamiltonian(ABC):
    def __init__(self, space: DiscreteSpace):
        self.space = space

    @abstractmethod
    def hamiltonian_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def _kinetic_matrix(self) -> np.ndarray:
        pass

    @abstractmethod
    def _potential_matrix(self) -> np.ndarray:
        pass


class SingleParticle(Hamiltonian):
    def __init__(self, space: DiscreteSpace, potential: Callable[[np.ndarray], np.ndarray]):

        super().__init__(space)
        self.potential = potential

    def hamiltonian_matrix(self):
        return self._kinetic_matrix() + self._potential_matrix()

    def _kinetic_matrix(self):

        eyes = [eye(n, dtype=self.space.dtype) for n in self.space.grid]
        Ts = [(diags([-1,-1, 2, -1,-1],
                     [-(self.space.grid[i]-1),-1, 0, 1,self.space.grid[i]-1],
                     shape=(self.space.grid[i], self.space.grid[i]),
                     dtype=self.space.dtype)
               / 2 / self.space.steps[i] ** 2).astype(self.space.dtype)
              for i in range(self.space.dim)]



        if self.space.dim == 1:
            return Ts[0]
        elif self.space.dim == 2:
            return kron(eyes[0], Ts[1]) + kron(Ts[0], eyes[1])
        elif self.space.dim == 3:
            return kron(eyes[0], kron(eyes[1], Ts[2])) + \
                   kron(eyes[0], kron(Ts[1], eyes[2])) + \
                   kron(Ts[0], kron(eyes[1], eyes[2]))

    def _potential_matrix(self):
        V: np.ndarray = self.potential(*self.space.grid_points)
        V = V.reshape(np.prod(self.space.grid))
        V = diags([V], [0], dtype=self.space.dtype)
        return V


class Solver:
    implemented_solvers = ('eigsh', 'lobpcg')

    def __init__(self, method: str = 'eigsh'):
        if method not in self.implemented_solvers:
            raise NotImplementedError(f"{method} solver is not available. Use one of {self.implemented_solvers}.")

        self.method = method

    def eigsys(self, hamiltonian: Hamiltonian, k: int) -> Tuple[np.ndarray, np.ndarray]:
        h_mat = hamiltonian.hamiltonian_matrix()

        if self.method == 'eigsh':
            eig, vec = eigsh(h_mat, k=k, which='LM',
                             sigma=0)  # use shift-invert to find smallest eigenvalues quickly
        elif self.method == 'lobpcg':
            # preconditioning matrix should approximate the inverse of the hamiltonian
            # we naively construct this by taking the inverse of diagonal elements
            # and setting all others to zero. This is called the Jacobi or diagonal preconditioner.
            A = diags([1 / h_mat.diagonal()], [0], dtype=hamiltonian.space.dtype).tocsc()
            precond = lambda x: A @ x
            M = LinearOperator(h_mat.shape, matvec=precond, matmat=precond, dtype=hamiltonian.space.dtype)

            # guess for eigenvectors is also computed from random numbers
            X_approx = np.random.rand(np.prod(hamiltonian.space.grid), k)

            sol = lobpcg(h_mat, X_approx, largest=False, M=M, tol=1e-15)
            eig, vec = sol[0], sol[1]
        else:
            raise NotImplementedError(
                f"{self.method} solver has not been implemented. Use one of {self.implemented_solvers}")

        return eig, vec.T


class Propagator:
    def __init__(self, hamiltonian: Hamiltonian,
                 init_state: Callable[[np.ndarray], np.ndarray],
                 solver: Solver, k: int):
        self.hamiltonian = hamiltonian
        self.solver = solver
        self.k = k
        self.init_state = init_state
        self.eigs, self.vecs = self.solver.eigsys(self.hamiltonian, self.k)

        # compute the initial state function on the discrete grid
        self.init_state_grid = self.init_state(*self.hamiltonian.space.grid_points)
        self.init_state_grid = self.hamiltonian.space.mesh_to_vec(self.init_state_grid)

    def evolve(self, times: np.ndarray):

        # compute the time-dependent wave function
        coeffs = self.vecs @ self.init_state_grid.T
        phases = np.exp(-1.0j * np.outer(times, self.eigs))
        coeffs_of_t = phases * coeffs.T
        psi_of_t = coeffs_of_t @ self.vecs

        # re-ravel the state vector to a mesh
        psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)

        return psi_of_t


if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt

    # define test parameters
    dim = 2
    support = (-8, 8)
    grid = 200
    dtype = np.float64
    num_states = 100
    potential_2d = lambda x, y: 1 / 2 * np.add.outer(x ** 2, 5*y**2) + np.outer(np.abs(x), np.abs(y))
    method = 'eigsh'
    init_state = lambda x, y: 1 / np.pi ** 0.25 * np.outer(
        np.exp(1.j * (-3) * x - 1 / 2 * (x-0) ** 2),
        np.exp(1.j * (0) * y - 1 / 2 * (y-0) ** 2)
    )

    space = DiscreteSpace(dim, support, grid, dtype)
    ham = SingleParticle(space, potential_2d)
    solver = Solver(method=method)
    prop = Propagator(ham, solver, num_states)
    print(prop.eigs)

    psit = prop.evolve(init_state, np.linspace(0,10,50))
    probt = np.abs(psit)**2

    for i, p in enumerate(probt):
        plt.matshow(p)
        plt.savefig(f"../assets/evolve_t{str(i).zfill(3)}.png")


