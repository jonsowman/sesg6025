# SESG6025 Gauss-Seidel SOR and Red-Black methods
# Jon Sowman 2013
from pprint import pprint
import numpy as np

def redblack(A, b, iterations=25, x=None):
    """
    Solve the linear matrix equation Ax = b via the Gauss Seidel 
    iterative method
    """
    # Get the number of elements in x
    n = len(A[0])
    
    # Create an initial guess
    if x is None:
        x = np.zeros(n)

    # Iterate 'iterations' times, should really check for convergence and 
    # quit early if so
    for its in range(iterations):
        for sweep in ('red', 'black'):
            print("sweeping %s" % sweep)
            for i in range(n):
                start = i % 2 if sweep == 'red' else 1 - i % 2
                t = 0
                for j in range(start, n, 2):
                    print("i=%d, j=%d" % (i, j))
                    if i == j:
                        continue
                    t = t + A[i, j] * x[j]
                x[i] = 1/A[i, i] * (b[i] - t)

    return x

def sor(A, b, iterations=25, x=None, omega=1.0):
    """
    Solve the linear matrix equation Ax = b via the (successive
    over relaxation (SOR) iterative method
    """
    # Create an initial guess
    if x is None:
        x = np.zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    D = np.diag(A)
    # and subtract them from A
    R = A - np.diagflat(D)

    # Get the number of elements in x
    n = len(A[0])

    # Iterate 'iterations' times
    for its in range(iterations):
        for i in range(n):
            t1 = 0
            t2 = 0
            for j in range(i):
                t1 = t1 + A[i, j] * x[j]
            for j in range(i+1, n):
                t2 = t2 + A[i, j] * x[j]
            x[i] = omega/A[i, i] * (b[i] - t1 - t2) + (1.0-omega)*x[i]
    return x

# Set up problem here
A = np.array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
b = np.array([7.0, -21.0, 15.0])
guess = np.array([1.0, 2.0, 3.0])

# Solve
rbsol = redblack(A, b, iterations=75, x=guess)
print("Red-Black solution is:")
pprint(rbsol)
assert np.allclose(rbsol, np.array([2.0, 4.0, 3.0]), atol = 1e-08)
