# SESG6025 Gauss-Seidel SOR and Red-Black methods
# Jon Sowman 2013
from pprint import pprint
import numpy as np

def rbidx(i, j, n):
    """
    Calculate the idx value for a red-black node at grid coordinates (i, j) 
    given that top left is always red.
    """
    idx = (i*n+j+1)/2 + (n**2)/2 if (i+j)%2 else (i*n+j)/2
    return idx

def rbstencil(n):
    """
    Construct a stencil for the red-black ordering of the adjacency matrix
    of size n^2.
    """
    A = -4 * np.eye(n**2)

    for i in range(n):
        for j in range(n):
            # Find the idx of this node
            idx = rbidx(i, j, n)
            # Neighbours are black (racist!)
            north = rbidx(i-1, j, n)
            south = rbidx(i+1, j, n)
            east = rbidx(i, j+1, n)
            west = rbidx(i, j-1, n)

            # Write to the stencil as long as the node exists
            if i > 0:
                A[idx, north] = 1
            if i < n - 1:
                A[idx, south] = 1
            if j > 0:
                A[idx, west] = 1
            if j < n - 1:
                A[idx, east] = 1

    return A

def gaussseidel(A, b, iterations=25, x=None):
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
        for i in range(n):
            t = 0
            for j in range(n):
                if i == j:
                    continue
                t = t + A[i, j] * x[j]
            tmp = 1/A[i, i] * (b[i] - t)
            print("x[%d] = %f" % (i, tmp))
            x[i] = tmp

        pprint(x)

    return x

def redblack(A, b, x=None, iterations=25):
    """
    Run a full red-black solution using the Gauss-Seidel method, i.e.
    solving black using the previously computed red solution vector
    """
    if x is None:
        x = np.zeros(len(A[0]))

    for i in range(iterations):
        x = rbstep(A, b, x)

    return x

def rbstep(A, b, x):
    """
    Solve the linear matrix equation Ax=b in a red-black manner that
    could be parallelised
    """

    n = len(A[0])
    c = n/2 + 1

    # Set up the matrices for red-black solving
    Dr = A[0:c, 0:c]
    Db = A[c:n, c:n]
    E = A[c:n, 0:c]
    F = E.transpose()

    # Split b into red and black parts
    br = b[0:c]
    bb = b[c:len(b)]

    # Split the existing solution into red and black parts
    xr = x[0:c]
    xb = x[c:len(x)]

    # Solve red
    xr = np.linalg.inv(Dr).dot(br - F.dot(xb))

    # Solve for black using the red x-values we just calculated
    xb = np.linalg.inv(Db).dot(bb - E.dot(xr))

    # Concatenate the solution vectors xr and xb and return
    return np.concatenate([xr, xb])

def gsstep(A, b, x):
    """
    Run one Gauss-Seidel step
    """
    
    n = len(A[0])

    for i in range(n):
        t = 0
        for j in range(n):
            if i == j:
                continue
            t = t + A[i, j] * x[j]
        tmp = 1/A[i, i] * (b[i] - t)
        x[i] = tmp

    return x

# Set up problem here
n = 3

#A_rep = 'np.array([[-4., -0., -0., -0., -0.],\n       [-0., -4., -0., -0.,\
#-0.],\n [-0., -0., -4., -0., -0.],\n       [-0., -0., -0., -4., -0.],\n \
#[-0.,-0., -0., -0., -4.]])'
#b_rep = 'np.array([ 0.  ,  0.  ,  0.08,  0.  ,  0.  ])'

A = rbstencil(n)
b = np.zeros([n**2])
b[(n**2)/4] = 0.08

# Solve
x = redblack(A, b, iterations=25)
print("Solution is:")
pprint(x)

