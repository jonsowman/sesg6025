##
# SESG6025 PDE Courswork
# Jon Sowman 2013
#
import numpy
import scipy
import argparse

def verify(s, h, exno, complex=False):
    """
    Verify that the solution matrix 's' meets the requirement
    that the Laplace equation is equation to 2 at coordinates (0.5, 0.5)
    with discretisation distance 'h'
    """
    tol = 1e-02
    # We will assume that we have a square matrix
    try:
        assert s.shape[0] == s.shape[1]
    except AssertionError:
       print("Solution matrix is not square")
    size = s.shape[0]
    c = (size - 1)/2

    # Estimate d2u/dx2 + d2u/dy2 using the appropriate stencil
    if complex:
        x = 16*s[c-1,c] - 30*s[c,c] + 16*s[c+1,c]
        y = 16*s[c,c-1] - 30*s[c,c] + 16*s[c,c+1] 
        if c-2 in range(n):
            x = x - s[c-2,c]
            y = y - s[c,c-2]
        if c+2 in range(n):
            x = x - s[c+2,c]
            y = y - s[c,c+2]
        x = x / (12*(h**2))
        y = y / (12*(h**2))
    else:
        x = (s[c-1,c] - 2*s[c,c] + s[c+1,c] ) / h**2
        y = (s[c,c-1] - 2*s[c,c] + s[c,c+1] ) / h**2
    result = x + y

    try:
        assert numpy.allclose(result, 2.0, atol=tol)
    except AssertionError:
        print('[ERROR Ex'+str(exno)+'] Solution verification failed: ' + \
                ' %.3f', result)
    else:
        print('[Ex' + str(exno) + '] ' + u'\u2207\u00b2' + \
                'u = 2 within ' + u'\u00b1' + str(tol) + \
                ' at u = (0.5,0.5) as required')


def sor(A, b, iterations=25, x=None, omega=1.0):
    """
    Solve the linear matrix equation Ax = b via the (successive
    over relaxation (SOR) iterative method. 'omega' is the relaxation
    parameter, which defaults to 1 (i.e. normal Gauss-Seidel)
    """
    # Create an initial guess
    if x is None:
        x = numpy.zeros(len(A[0]))

    # Create a vector of the diagonal elements of A
    D = numpy.diag(A)
    # and subtract them from A
    R = A - numpy.diagflat(D)

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

def embed(a, value):
    # Embed matrix into an array with the boundary conditions in
    # a is the matrix and value is the value on the (fixed) boundary
    size = a.shape[0]
    a_tmp = numpy.zeros([size+2, size+2])

    for i in range(1, size+1):
        for j in range(1, size+1):
            a_tmp[i, j] = a[i-1, j-1]

    return a_tmp

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
    A = -4 * numpy.eye(n**2)

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

def redblack(A, b, x=None, iterations=25):
    """
    Run a full red-black solution using the Gauss-Seidel method, i.e.
    solving black using the previously computed red solution vector
    """
    if x is None:
        x = numpy.zeros(len(A[0]))

    for i in range(iterations):
        x = rbstep(A, b, x)

    # Reorder the solution vector
    c = x.size/2
    sol = numpy.empty(x.size, dtype=x.dtype)
    sol[0::2] = x[0:c+1]
    sol[1::2] = x[c+1:len(x)]
    return sol

def rbstep(A, b, x):
    """
    Run one iteration of the Red-Black Gauss-Seidel solver. This is
    not run in a parallel manner but could be if required.
    Note that the solution is completely parallel within the red
    solving, and ditto for the black. 
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
    xr = numpy.linalg.inv(Dr).dot(br - F.dot(xb))

    # Solve for black using the red x-values we just calculated
    xb = numpy.linalg.inv(Db).dot(bb - E.dot(xr))

    # Concatenate the solution vectors xr and xb and return
    return numpy.concatenate([xr, xb])

def complex_stencil(n):
    """
    Use the more complex 8 point stencil described in the coursework handout
    to construct the matrix a
    """

    a = numpy.zeros([n**2, n**2])

    for i in range(n):
        for j in range(n):
            idx = i*n+j
            idx_n = (i-1)*n+j
            idx_nn = (i-2)*n+j
            idx_s = (i+1)*n+j
            idx_ss = (i+2)*n+j
            idx_e = i*n+j+1
            idx_ee = i*n+j+2
            idx_w = i*n+j-1
            idx_ww = i*n+j-2

            # Central index will always exist in the matrix
            a[idx, idx] = -60
            if (idx_nn-j)/n in range(n):
                a[idx, idx_nn] = -1
            if (idx_n-j)/n in range(n):
                a[idx, idx_n] = 16
            if (idx_ss-j)/n in range(n):
                a[idx, idx_ss] = -1
            if (idx_s-j)/n in range(n):
                a[idx, idx_s] = 16
            if idx_ee in range(i*n, (i+1)*n):
                a[idx, idx_ee] = -1
            if idx_e in range(i*n, (i+1)*n):
                a[idx, idx_e] = 16
            if idx_ww in range(i*n, (i+1)*n): 
                a[idx, idx_ww] = -1
            if idx_w in range(i*n, (i+1)*n):
                a[idx, idx_w] = 16

    return a

def simple_stencil(n, debug):
    """
    Use the stencil derived in lectures, the first central difference
    approximation
    """

    # Clear matrix and set it up
    a = numpy.zeros([n**2, n**2])

    if(debug):
        print('================')
        print('Interior')

    # Build full matrix
    # Interior
    for i in range(1, n-1):
        for j in range(1, n-1):
            north = (i-1)*n+j
            west = i*n+j-1
            index = i*n+j
            east = i*n+j+1
            south = (i+1)*n+j

            a[index, north] = 1
            a[index, west] = 1
            a[index, index] = -4
            a[index, east] = 1
            a[index, south] = 1

    if debug:
        print(a)

        # Edges
        # North/Top (nothing further North)
        print('================')
        print("Top")

    # First row number
    i = 0

    # Note that the range(1, n-1) means that we JUST middle ones
    # e.g. if n=5 then range(1, 4) = [1, 2, 3]
    for j in range(1, n-1):
        #north = (i-1)*n+j
        west = i*n+j-1
        index = i*n+j
        east = i*n+j+1
        south = (i+1)*n+j

        #a[index, north] = 1
        a[index, west] = 1
        a[index, index] = -4
        a[index, east] = 1
        a[index, south] = 1

    # West/Left (nothing further West)
    if debug:
        print('================')
        print("West")

    # First column number
    j = 0

    for i in range(1, n-1):
        north = (i-1)*n+j
        #west = i*n+j-1
        index = i*n+j
        east = i*n+j+1
        south = (i+1)*n+j

        a[index, north] = 1
        #a[index, west] = 1
        a[index, index] = -4
        a[index, east] = 1
        a[index, south] = 1

    # East/Right (nothing further East)
    if debug:
        print('================')
        print("East")

    # Last column number
    j = n - 1

    for i in range(1, n-1):
        north = (i-1)*n+j
        west = i*n+j-1
        index = i*n+j
        #east = i*n+j+1
        south = (i+1)*n+j

        a[index, north] = 1
        a[index, west] = 1
        a[index, index] = -4
        #a[index, east] = 1
        a[index, south] = 1

    # South / Bottom (nothing further South)
    if debug:
        print('================')
        print("South")

    # Last row number
    i = n - 1

    for j in range(1, n-1):
        north = (i-1)*n+j
        west = i*n+j-1
        index = i*n+j
        east = i*n+j+1
        #south = (i+1)*n+j

        a[index, north] = 1
        a[index, west] = 1
        a[index, index] = -4
        a[index, east] = 1
        #a[index, south] = 1

    if debug:
        print('================')
        print("Corners")

    # Top Left
    i = 0
    j = 0
    index = i*n+j
    east = i*n+j+1
    south = (i+1)*n+j

    a[index, index] = -4
    a[index, east] = 1
    a[index, south] = 1

    # Top Right
    i = 0
    j = n - 1
    index = i*n+j
    west = i*n+j-1
    south = (i+1)*n+j

    a[index, index] = -4
    a[index, west] = 1
    a[index, south] = 1

    # Bottom Left
    i = n - 1
    j = 0
    index = i*n+j
    north = (i-1)*n+j
    east = i*n+j+1

    a[index, index] = -4
    a[index, north] = 1
    a[index, east] = 1

    # Bottom Right
    i = n - 1
    j = n - 1
    index = i*n+j
    north = (i-1)*n+j
    west = i*n+j-1

    a[index, index] = -4
    a[index, north] = 1
    a[index, west] = 1

    return a

# Parse the command line options
parser = argparse.ArgumentParser(description="SESG6025 Coursework by \
        Jon Sowman. PDE solver using various methods")
parser.add_argument("-d", "--debug", action="store_true", \
        help="Show debug output from program")
parser.add_argument("-p", "--plot", action="store_true", \
        help="Plot the solution")
parser.add_argument("-n", type=int, help="Size of the grid (nxn), \
        defaults to 3")
parser.add_argument("-v", "--verbosity", type=int, help="Verbosity level \
        from 1-3 inclusive")
args = parser.parse_args()

# Set up printing of the array so it displays nicely
numpy.set_printoptions(precision=0, linewidth=120)

# n is the size of the mesh with the unknowns in it
# So the matrix will be of size n+2
if args.n:
    n = args.n
else:
    n = 3
n_full = n + 2

# The h value is 1/(n+2) : taking into account the intervals
# to get to the boundary
h = 1.0 / n_full

# Compute the matrices for simple and complex stencils
if args.verbosity >= 1:
    print("Setting up simple stencil"),
a_simple = simple_stencil(n, args.debug)
if args.verbosity >= 1:
    print("...done")
    if args.verbosity >= 2:
        print("Simple stencil is")
        print(a_simple)

if args.verbosity >= 1:
    print("Setting up complex stencil"),
a_complex = complex_stencil(n)
if args.verbosity >= 1:
    print("...done")
    if args.verbosity >= 2:
        print("Complex stencil is")
        print(a_complex)

if args.verbosity >= 1:
    print("Setting up red-black stencil"),
a_redblack = rbstencil(n)
if args.verbosity >= 1:
    print("...done")
    if args.verbosity >= 2:
        print("Red-black stencil is")
        print(a_redblack)

# Reset printing options
numpy.set_printoptions()
numpy.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
        precision=8, suppress=False, threshold=1000)

# Example 3: 2D heat equation in steady state
# i.e. the Laplace equation with boundary conditions
if args.debug:
    print('================')
b_simple = numpy.zeros([n**2, 1])
b_complex = numpy.zeros([n**2, 1])
b_redblack = numpy.zeros([n**2])

# For rho(0.5, 0.5) = 2 we just require that the middle element of b
# is -2*(h**2), as per equation 4.51 for the simple stencil.
# For the complex stencil there's a factor of 12 in there too
# For the red-black solver, the linear system of equations is in a different
# order
b_simple[((n**2)-1)/2] = 2*(h**2)
b_complex[((n**2)-1)/2] = 12*2*(h**2)
b_redblack[(n**2)/4] = 2*(h**2)

if args.verbosity >= 2:
    print("b_simple is:")
    print(b_simple)
    print("b_complex is:")
    print(b_complex)
    print("b_redblack is:")
    print(b_redblack)

# For ex1 we use the np method, for ex2 we use our own SOR method
if args.verbosity >= 1:
    print("Solving with numpy"),
ex1_soln = numpy.linalg.solve(a_simple, b_simple)
if args.verbosity >= 1:
    print("...done")
    print("Solving with Gauss-Seidel"),
    if args.verbosity >= 2:
        print("Solution is:")
        print(ex1_soln)
ex2_soln = sor(a_simple, b_simple, iterations=50)
if args.verbosity >= 1:
    print("...done")

# Now let's solve the complex stencil problem
if args.verbosity >= 1:
    print("Solving complex stencil using numpy"),
ex3_soln = numpy.linalg.solve(a_complex, b_complex)
if args.verbosity >= 1:
    print("...done")

# Finally, solve the red-black problem
if args.verbosity >= 1:
    print("Solving simple stencil in Red-Black formulation"),
ex4_soln = redblack(a_redblack, b_redblack, iterations=25)
if args.verbosity >= 1:
    print("...done")

if args.debug:
    raw_input("Press return to continue")
    print("")

# Wrap the solution onto grid and embed
ex1_full = embed(numpy.reshape(ex1_soln, [n, n]), 0)
ex2_full = embed(numpy.reshape(ex2_soln, [n, n]), 0)
ex3_full = embed(numpy.reshape(ex3_soln, [n, n]), 0)
ex4_full = embed(numpy.reshape(ex4_soln, [n, n]), 0)

if args.debug:
    print('================')
if args.verbosity >= 1:
    print("Exercise 1: Solve PDE using numpy QR solver and simple stencil")
verify(ex1_full, h, 1)
if args.verbosity >= 1:
    print("Exercise 2: Solve PDE using SOR solver and simple stencil")
verify(ex2_full, h, 2)
if args.verbosity >= 1:
    print("Exercise 3: Solve PDE using numpy QR solver and complex stencil")
verify(ex3_full, h, 3, complex=True)
if args.verbosity >= 1:
    print("Exercise 4: Solve PDE using Gauss-Seidel Red/Black solver " + \
            "and simple stencil")
verify(ex4_full, h, 4)

if args.debug:
    print('================')
    print("Plotting...")

# 3D plotting part
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

steps = 2.0 + n

h = 1.0 / (steps - 1)
if args.debug:
    print('h = ', h)
X = np.arange(0, steps, 1) * h
if args.debug:
    print(X)
    print
Y = np.arange(0, steps, 1) * h
X, Y = np.meshgrid(X, Y)

#print("X is: ")
#print(X)

#print("Y is: ")
#print(Y)

#R = np.sqrt(X**2 + Y**2)
#surf = ax.plot_wireframe(X, Y, R, rstride=1, cstride=1)
#surf = ax.plot_wireframe(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm, 
#    linewidth=0, antialiased=False, shade=True)
surf = ax.plot_wireframe(X, Y, ex4_full, rstride=1, cstride=1)

if args.plot:
    plt.show()
