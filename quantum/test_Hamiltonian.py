import numpy as np
from hamiltonian import SingleParticle, DiscreteSpace


def test_solver():
    import time

    # define test parameters
    n_eigs = 10
    support = (-10, 10)
    dtype = np.float64

    potential_1d = lambda x: 1 / 2 * x ** 2
    opotential_2d = lambda x, y: 1 / 2 * np.add.outer(x ** 2, y ** 2)
    potential_3d = lambda x, y, z: 1 / 2 * np.add.outer(np.add.outer(x ** 2, y ** 2), z ** 2)

    test_suite_1 = [dict(dim=1, v=potential_1d, grid=1000, solver='eigsh'),
                    dict(dim=1, v=potential_1d, grid=1000, solver='lobpcg'),
                    dict(dim=2, v=potential_2d, grid=200, solver='eigsh'),
                    dict(dim=2, v=potential_2d, grid=200, solver='lobpcg'),
                    dict(dim=3, v=potential_3d, grid=25, solver='eigsh'),
                    dict(dim=3, v=potential_3d, grid=25, solver='lobpcg')]

    # run tests
    for t in test_suite_1:
        print("-" * 20)
        print("Solving with", t)
        dim = t['dim']
        v = t['v']
        grid = t['grid']
        solver = t['solver']

        space = DiscreteSpace(dim, support, grid, dtype)
        ham = SingleParticle(space, v, solver=solver)
        ti = time.time()
        eigs, vecs = ham.solve(n_eigs)
        tf = time.time()
        print(f"{tf - ti:0.3} seconds to solve.")
        print("Eigenvals:", eigs)
