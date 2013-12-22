# SESG6025 Jacobi Method
# Jon Sowman 2013
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot

def jacobi(A, b, N=25, x=None):
    """
    Solve the linear matrix equation Ax = b via the Jacobi iterative method
    """
    # Create an initial guess
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    D = diag(A)
    # and subtract them from A
    R = A - diagflat(D)

    # Iterate for N times
    for i in range(N):
        x = (b - dot(R, x))/D
        pprint(x)
    return x

# Set up problem here
A = array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
b = array([7.0, -21.0, 15.0])
guess = array([1.0, 2.0, 3.0])

# Solve
sol = jacobi(A, b, N=25, x=guess)
print("A: ")
pprint(A)

print("b: ")
pprint(b)

print("x: ")
pprint(sol)
