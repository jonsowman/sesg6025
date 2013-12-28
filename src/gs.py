# SESG6025 Gauss-Seidel SOR Method
# Jon Sowman 2013
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot, allclose, linalg, empty

def gaussseidel(A, b, iterations=25, x=None):
    """
    Solve the linear matrix equation Ax = b via the Gauss Seidel 
    iterative method
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
            t = 0
            for j in range(n):
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
            x[i] = omega/A[i, i] * (b[i] - t1 - t2) + (1.0-omega)*x[i]
    return x

def redblack(A, b, iterations=25, x=None):
    """
    Solve the linear matrix equation Ax = b via the Gauss Seidel Red-Black
    method and the solver of our choice.
    """

    debug = True

    # First node is always red (u0), so let's construct A_red and A_black
    A_red = A[::2]
    A_blk = A[1::2]
    print("A_red is:")
    pprint(A_red)
    print("A_blk is:")
    pprint(A_blk)

    # Now we must divide up b
    b_red = b[::2]
    b_blk = b[1::2]

    # And finally, x (if it exists)
    if x is not None:
        x_red = x[::2]
        x_blk = x[1::2]

    # Solve the two matrix equations
    x_red = linalg.solve(A_red, b_red)
    x_blk = linalg.solve(A_blk, b_blk)

    # Recombine the red & black solutions to produce the result
    x_res = empty(A[0].size, A.dtype)
    return x_res

# Set up problem here
A = array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
b = array([7.0, -21.0, 15.0])
guess = array([1.0, 2.0, 3.0])

# Solve
sol = gaussseidel(A, b, iterations=25, x=guess)
print("Normal Gauss Seidel solution is:")
pprint(sol)
assert allclose(sol, array([2.0, 4.0, 3.0]), atol = 1e-08)

rbsol = redblack(A, b, iterations=25, x=guess)
print("Red-Black solution is:")
pprint(rbsol)
assert allclose(rbsol, array([2.0, 4.0, 3.0]), atol = 1e-08)
