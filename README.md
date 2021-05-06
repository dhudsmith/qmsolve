# qmsolve

This package allows you to quickly and accuractely solve the single-particle Schrodinger equation 
in 1, 2, or 3D. 

For example, let's look at the cannonical 1D harmonic oscillator: 
```python3
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

#>>> [0.49979912 1.49899526 2.49738674 3.49497257 4.49175179]
```

How about in 3D? 
```python3
from hamiltonian import DiscreteSpace, SingleParticle, Solver

potential = lambda x, y, z: 1/2 * (x[None, None]**2 + 
                                   y[None, :, None]**2 + 
                                   z[:, None, None]**2)

# setup the space, hamiltonian, and solver
# use the lobpcg solver this time for faster convergence
space = DiscreteSpace(dim=3, support=(-10,10), grid=50)
hamil = SingleParticle(space, potential)
solver = Solver(method="lobpcg")

# get the first few eigenstates
eigs, _ = solver.eigsys(hamil, 5)
print(eigs)

#>>> [1.48421365 2.46296504 2.46300386 2.46634141 3.42413886]
#compare w/ [1.5, 2.5, 2.5, 2.5, 3.5]
```

 
