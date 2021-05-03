from typing import Callable
from abc import abstractmethod, ABC

import numpy as np

from hamiltonian import Hamiltonian, Solver

class Propagator(ABC):
    def __init__(self, hamiltonian: Hamiltonian, init_state: Callable[[np.ndarray], np.ndarray]):
        self.hamiltonian = hamiltonian
        self.init_state = init_state

        # compute the initial state function on the discrete grid
        self.init_state_grid = self.init_state(*self.hamiltonian.space.grid_points)
        self.init_state_grid = self.hamiltonian.space.mesh_to_vec(self.init_state_grid)

    @abstractmethod
    def evolve(self, times: np.ndarray) -> np.ndarray:
        pass


class VisscherPropagator(Propagator):
    def __init__(self, hamiltonian: Hamiltonian, init_state: Callable[[np.ndarray], np.ndarray]):
        super().__init__(hamiltonian, init_state)

        self.hmat = self.hamiltonian.hamiltonian_matrix()

    def evolve(self, times: np.ndarray) -> np.ndarray:

        # time params:
        numt = len(times)
        dt = times[1] - times[0]
        if dt==0:
            raise ValueError("Timestep must be constant and greater than zero.")

        # allocate wf matrices
        numx = np.prod(self.hamiltonian.space.grid)
        R_all = np.zeros((numt, numx), dtype=self.hamiltonian.space.dtype)
        I_all = R_all.copy()

        # compute initial condition
        psi0 = self.init_state_grid.T.squeeze()
        # take a half time step forward to compute initial imaginary part
        psi1 = psi0 + 1.j * dt/2 * self.hmat @ psi0 - (dt/2)**2 / 2 * self.hmat @ self.hmat @ psi0
        R_all[0] = psi0.real
        I_all[0] = psi1.imag

        # loop over time
        for ix in range(1, numt):
            R_all[ix], I_all[ix] = self.visscher_step(R_all[ix-1], I_all[ix-1], dt)

            if ix % 999 == 0:
                print(f"Taking step {ix} of {numt - 1}.", R_all[ix].mean(), I_all[ix].mean())

        # take a half step backward to align time grids
        _, I_all = self.visscher_step(R_all, I_all, -dt/2)

        psi_of_t = R_all + 1.j * I_all
        psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)
        return psi_of_t


    def visscher_step(self, R, I, dt):
        Rnext = R + dt * I @ self.hmat
        Inext = I - dt * R @ self.hmat

        return Rnext, Inext





class DiagonalizationPropagator(Propagator):
    def __init__(self, hamiltonian: Hamiltonian, init_state: Callable[[np.ndarray], np.ndarray], solver: Solver,
                 k: int):
        super().__init__(hamiltonian, init_state)

        self.solver = solver
        self.k = k
        self.eigs, self.vecs = self.solver.eigsys(self.hamiltonian, self.k)

    def evolve(self, times: np.ndarray) -> np.ndarray:

        # compute the time-dependent wave function
        coeffs = self.vecs @ self.init_state_grid.T
        phases = np.exp(-1.0j * np.outer(times, self.eigs))
        coeffs_of_t = phases * coeffs.T
        psi_of_t = coeffs_of_t @ self.vecs

        # re-ravel the state vector to a mesh
        psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)

        return psi_of_t
