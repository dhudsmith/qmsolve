from typing import Callable, Generator
from abc import abstractmethod, ABC

import numpy as np

from hamiltonian import Hamiltonian, Solver


class Propagator(ABC):
    """
    The abstract base class for time evolution. Child classes should implement the `evolve` method
    which returns a generator object for the time-dependent wave function.
    """
    def __init__(self, hamiltonian: Hamiltonian,
                 init_state: Callable[[np.ndarray], np.ndarray],
                 dt: float):

        if dt==0:
            raise ValueError("Timestep dt must not be zero.")

        self.hamiltonian = hamiltonian
        self.init_state = init_state
        self.dt = dt

        # compute the initial state function on the discrete grid
        self.init_state_grid = self.init_state(*self.hamiltonian.space.grid_points)
        self.init_state_grid = self.hamiltonian.space.mesh_to_vec(self.init_state_grid)

    @abstractmethod
    def evolve(self) -> Generator[np.ndarray, None, None]:
        pass


class VisscherPropagator(Propagator):
    """
    Time evolution based on explicit time integration using Visscher's method.
    "A fast explicit algorithm for the time‐dependent Schrödinger equation"
    Computers in Physics 5, 596 (1991); https://doi.org/10.1063/1.168415.
    """
    def __init__(self, hamiltonian: Hamiltonian,
                 init_state: Callable[[np.ndarray], np.ndarray],
                 dt: float,
                 sample_interval: int):
        """
        Construct a Visscher Propagator. This Propagator takes an additional `sample_interval`
        argument to allow sampling of wave functions at longer time intervals than used for solving.
        used for integrating the TDSE.
        :param hamiltonian: The hamiltonian for the system.
        :param init_state: The initial state function
        :param dt: The time step to use when integrating the TDSE
        :param sample_interval: The number of time steps between output wave functions.
        """
        super().__init__(hamiltonian, init_state, dt)

        self.sample_interval = sample_interval
        self.hmat = self.hamiltonian.hamiltonian_matrix()

    def evolve(self) -> Generator[np.ndarray, None, None]:
        """
        Generate the time evolved wave function.
        :return: a generator object which can be used to fetch the wave function values
        at successive timesteps.
        """
        psi0 = self.init_state_grid.T.squeeze()
        psi1 = psi0 + 1.j * (self.dt/2) * self.hmat @ psi0
        R = psi0.real
        I = psi1.imag

        # loop over time
        step_num = 0
        while True:
            Rnext = R + self.dt * I @ self.hmat
            Inext = I - self.dt * Rnext @ self.hmat

            if step_num % self.sample_interval == 0:
                # yield a sample. Approximate imaginary part at t as average of t-1/2dt and t+1/2dt
                psi_of_t = Rnext + 1.j/2*(I+Inext)
                psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)

                yield psi_of_t.squeeze()

            R, I = Rnext, Inext
            step_num += 1


class DiagonalizationPropagator(Propagator):
    """
    Time evolution based on exact diagonalization of the Hamiltonian.
    """
    def __init__(self, hamiltonian: Hamiltonian,
                 init_state: Callable[[np.ndarray], np.ndarray],
                 dt: float,
                 solver: Solver,
                 k: int):
        """
        Construct a diagonalization-based propagator object. As this
        method relies upon diagonalization of the hamiltonian, you must pass an
        additional solver object.
        :param hamiltonian: The hamiltonian for the system.
        :param init_state: The initial state function.
        :param dt: The time step at which to sample the time-dependent wave function.
        :param solver: The solver object for diagonalizing the hamiltonian
        :param k: The number of eigenstates to use when approximating the exact propagator.
        """
        super().__init__(hamiltonian, init_state, dt)

        self.solver = solver
        self.k = k
        self.eigs, self.vecs = self.solver.eigsys(self.hamiltonian, self.k)

    def evolve(self) -> Generator[np.ndarray, None, None]:
        """
        Generate the time evolved wave function.
        :return: a generator object which can be used to fetch the wave function values
        at successive timesteps.
        """

        coeffs = self.vecs @ self.init_state_grid.T

        # loop over time
        step_num = 0
        t = 0
        while True:
            phase_t = np.exp(-1.0j * t * self.eigs)
            psi_of_t = phase_t * coeffs.T @ self.vecs
            psi_of_t = self.hamiltonian.space.vec_to_mesh(psi_of_t)

            yield psi_of_t.squeeze()

            step_num += 1
            t+=self.dt