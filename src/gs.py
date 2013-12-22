# SESG6025 Gauss-Seidel Method
# Jon Sowman 2013
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, allclose

def gs(A, b, iterations=25, x=None):
    """
    Solve the linear matrix equation Ax = b via the gs iterative method
    """
    # Create an initial guess
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    D = diag(A)
    # and subtract them from A
    R = A - diagflat(D)

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
            x[i] = 1/A[i, i] * (b[i] - t1 - t2)
    return x

# Set up problem here
A = array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
b = array([7.0, -21.0, 15.0])
guess = array([1.0, 2.0, 3.0])

# Solve
sol = gs(A, b, iterations=25, x=guess)
pprint(sol)
#assert allclose(sol, array([2.0, 4.0, 3.0]), atol = 1e-08)
