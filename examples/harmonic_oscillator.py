from hamiltonian import DiscreteSpace, SingleParticle, Solver

# harmonic oscillator potential
potential = lambda x: 1/2 * x**2

# setup the space, hamiltonian, and solver
space = DiscreteSpace(dim=1, support=(-20,20), grid=500)
hamil = SingleParticle(space, potential)
solver = Solver()

# get the first few eigenstates
eigs, _ = solver.eigsys(hamil, 5)
print(eigs)

#----------------------
# 3d
#----------------------

# harmonic oscillator potential
potential = lambda x, y, z: 1/2 * (x[None, None]**2 + y[None, :, None]**2 + z[:, None, None]**2)

# setup the space, hamiltonian, and solver
space = DiscreteSpace(dim=3, support=(-10,10), grid=50)
hamil = SingleParticle(space, potential)
solver = Solver(method='lobpcg')

# get the first few eigenstates
eigs, _ = solver.eigsys(hamil, 5)
print(eigs)