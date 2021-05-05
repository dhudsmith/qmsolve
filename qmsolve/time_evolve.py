from typing import Callable, Generator
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
    def evolve(self, dt: float, sample_interval: int) -> Generator[np.ndarray, None, None]:
        pass


class VisscherPropagator(Propagator):
    def __init__(self, hamiltonian: Hamiltonian, init_state: Callable[[np.ndarray], np.ndarray]):
        super().__init__(hamiltonian, init_state)

        self.hmat = self.hamiltonian.hamiltonian_matrix()

    def evolve(self, dt: float, sample_interval: int) -> Generator[np.ndarray, None, None]:

        # time params:
        if dt == 0:
            raise ValueError("Timestep must not be zero.")

        # compute initial conditions
        # take a half time step forward to compute initial imaginary part
        psi0 = self.init_state_grid.T.squeeze()
        psi1 = psi0 + 1.j * (dt/2) * self.hmat @ psi0
        R = psi0.real
        I = psi1.imag

        # loop over time
        step_num = 0
        while True:
            Rnext = R + dt * I @ self.hmat
            Inext = I - dt * Rnext @ self.hmat

            if step_num % sample_interval == 0:
                # yield a sample. Approximate imaginary part at t as average of t-1/2dt and t+1/2dt
                psi_of_t = Rnext + 1.j/2*(I+Inext)
                psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)

                yield psi_of_t.squeeze()

            R, I = Rnext, Inext
            step_num += 1


class DiagonalizationPropagator(Propagator):
    def __init__(self, hamiltonian: Hamiltonian, init_state: Callable[[np.ndarray], np.ndarray], solver: Solver,
                 k: int):
        super().__init__(hamiltonian, init_state)

        self.solver = solver
        self.k = k
        self.eigs, self.vecs = self.solver.eigsys(self.hamiltonian, self.k)

    def evolve(self, dt: float, sample_interval: int = 1) -> Generator[np.ndarray, None, None]:
        # compute the time-dependent wave function
        coeffs = self.vecs @ self.init_state_grid.T

        # loop over time
        step_num = 0
        t = 0
        while True:
            phase_t = np.exp(-1.0j * t * self.eigs)
            psi_of_t = phase_t * coeffs.T @ self.vecs

            if step_num % sample_interval == 0:
                psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)
                yield psi_of_t.squeeze()

            step_num += 1
            t+=dt