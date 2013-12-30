# SESG6025 Gauss-Seidel SOR and Red-Black methods
# Jon Sowman 2013
from pprint import pprint
import numpy as np

def rb(A, b, x=None):
    """
    Solve in a red-black manner
    """
    A_red = A[::2]
    A_blk = A[1::2]

    print("A_red is:")
    pprint(A_red)
    print("A_black is:")
    pprint(A_blk)
    
    b_red = b[::2]
    b_blk = b[1::2]

    print("b_red is:")
    pprint(b_red)
    print("b_black is:")
    pprint(b_blk)

    x_red = np.linalg.solve(A_red, b_red)
    x_blk = np.linalg.solve(A_blk, b_blk)

    print("x_red is:")
    pprint(x_red)
    print("x_black is:")
    pprint(x_blk)

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

    # Iterate 'iterations' times, should really check for 
    # convergence and quit early if so
    for its in range(iterations):
        for sweep in ('red', 'black'):
            print("Sweeping %s..." % sweep)
            for i in range(n):
                t = 0
                print("Working on row %d" % i)
                start = i % 2 if sweep == 'red' else 1 - i % 2
                for j in range(start, n, 2):
                    print("Working on %d,%d..." % (i, j)),
                    if i == j:
                        print("skipped")
                        continue
                    print("done")
                    t = t + A[i,j] * x[j]
                    print("t=%f" % t)
                tmp = 1/A[i,i] * (b[i] - t)
                print("setting x[%d]=%f" % (i, tmp))
                x[i] = tmp
            pprint(x)

    return x

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
            x[i] = tmp

    return x

# Set up problem here
#A = np.array([[4.0, -1.0, 1.0], [4.0, -8.0, 1.0], [-2.0, 1.0, 5.0]])
#b = np.array([7.0, -21.0, 15.0])
#guess = np.array([1.0, 2.0, 3.0])

A_rep = 'np.array([[-4., -0., -0., -0., -0.,  1.,  1., -0., -0.],\n       [-0., \
-4., -0., -0., -0.,  1.,  0.,  1., -0.],\n       [-0., -0., -4., -0., -0.,  1.,\
1.,  1.,  1.],\n       [-0., -0., -0., -4., -0., -0.,  1., -0.,  1.],\n\
[-0., -0., -0., -0., -4., -0., -0.,  1.,  1.],\n       [ 1.,  1.,  1., -0.,\
-0., -4., -0., -0., -0.],\n       [ 1., -0.,  1.,  1., -0., -0., -4., -0.,\
-0.],\n       [-0.,  1.,  1., -0.,  1., -0., -0., -4., -0.],\n       [-0., \
-0., 1.,  1.,  1., -0., -0., -0., -4.]])'
b_rep = 'np.array([ 0.  ,  0.  ,  0.08,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.\
])'
A = eval(A_rep)
b = eval(b_rep)

# Solve
sol = gaussseidel(A, b, iterations=1)
print("Normal solution is:")
pprint(sol)

#print("=================")

#rbsol = redblack(A, b, iterations=1)
#print("Red-Black solution is:")
#pprint(rbsol)
