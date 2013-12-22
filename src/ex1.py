##
# SESG6025 PDE Courswork
# Jon Sowman 2013
#
import numpy
import scipy

def verify(s, h):
    """
    Verify that the solution matrix 's' meets the requirement
    that the Laplace equation is equation to 2 at coordinates (0.5, 0.5)
    with discretisation distance 'h'
    """
    # We will assume that we have a square matrix
    if not s.shape[0] == s.shape[1]:
        raise AssertionError("Solution matrix is not square")
    size = s.shape[0]
    c = (size - 1)/2

    # Estimate d2u/dx2 + d2u/dy2
    x = (s[c-1,c] - 2*s[c,c] + s[c+1,c] ) / h**2
    y = (s[c,c-1] - 2*s[c,c] + s[c,c+1] ) / h**2
    result = x+y

    assert numpy.allclose(result, 2.0, atol=1e-08)
    print('[Ex1] ' + u'\u2207\u00b2' + 'u = 2 at u = (0.5,0.5) as required')


def sor(A, b, iterations=25, x=None, omega=1.0):
    """
    Solve the linear matrix equation Ax = b via the (successive
    over relaxation (SOR) iterative method
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

    #for i in range(0, size+2):
    #    a_tmp[0, i] = value
    return a_tmp

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
            if idx_nn in range(n**2):
                a[idx, idx_nn] = -1
            if idx_n in range(n**2):
                a[idx, idx_n] = 16
            if idx_ss in range(n**2):
                a[idx, idx_ss] = -1
            if idx_s in range(n**2):
                a[idx, idx_s] = 16
            if idx_ee in range(n**2):
                a[idx, idx_ee] = -1
            if idx_e in range(n**2):
                a[idx, idx_e] = 16
            if idx_ww in range(n**2):
                a[idx, idx_ww] = -1
            if idx_w in range(n**2):
                a[idx, idx_w] = 16

    return a

def simple_stencil(n):
    """
    Use the stencil derived in lectures, the first central difference
    approximation
    """
    # Clear matrix and set it up
    a = numpy.zeros([n**2, n**2])

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
            print i, j, index

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
        print i, j, index

    # West/Left (nothing further West)
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
        print i, j, index

    # East/Right (nothing further East)
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
        print i, j, index

    # South / Bottom (nothing further South)
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
        print i, j, index

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
    print i, j, index

    # Top Right
    i = 0
    j = n - 1
    index = i*n+j
    west = i*n+j-1
    south = (i+1)*n+j

    a[index, index] = -4
    a[index, west] = 1
    a[index, south] = 1
    print i, j, index

    # Bottom Left
    i = n - 1
    j = 0
    index = i*n+j
    north = (i-1)*n+j
    east = i*n+j+1

    a[index, index] = -4
    a[index, north] = 1
    a[index, east] = 1
    print i, j, index

    # Bottom Right
    i = n - 1
    j = n - 1
    index = i*n+j
    north = (i-1)*n+j
    west = i*n+j-1

    a[index, index] = -4
    a[index, north] = 1
    a[index, west] = 1
    print i, j, index

    return a

# Set up printing of the array so it displays nicely
numpy.set_printoptions(precision=0, linewidth=120)

# n is the size of the mesh with the unknowns in it
# So the matrix will be of size n+2
n = 3
n_full = n + 2

# The h value is 1/(n+2) : taking into account the intervals
# to get to the boundary
h = 1.0 / n_full
print("h is %.3f" % h)

# THIS IS THE FINAL MATRIX
a = complex_stencil(n)
print a
print('================')
raw_input("Press return to continue")
print("")

# Reset printing options
numpy.set_printoptions()
numpy.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
        precision=8, suppress=False, threshold=1000)

# Example 3: 2D heat equation in steady state
# i.e. the Laplace equation with boundary conditions
print('================')
print("Set up b")
b = numpy.zeros([n**2, 1])

# For rho(0.5, 0.5) = 2 we just require that the middle element of b
# is -2*(h**2), as per equation 4.51
b[((n**2)-1)/2] = 2*(h**2)

# This sets it up by hand as a check
#b = [[0], [0], [0], [0], [-2*(h**2)], [0], [0], [0], [0]]
#b = [[0.0], [-0.5], [0.0], [-0.5], [2.0], [-0.5], [0.0], [-0.5], [0.0]]

print(b)
print('================')
raw_input("Press return to continue")
print("")

print("In 2D with 9 unknowns, solution is:")
# Use Gauss-Seidel iterative method
soln = sor(a, b, iterations=50)

print(soln)
print('================')
raw_input("Press return to continue")
print("")

# Wrap the solution onto grid and embed
soln_wrap = numpy.reshape(soln, [n, n])
soln_full = embed(soln_wrap, 100)
print soln_full

print('================')
verify(soln_full, h)

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
print steps

h = 1.0 / (steps - 1)
print('h = ', h)
X = np.arange(0, steps, 1) * h
print(X)
print
Y = np.arange(0, steps, 1) * h
X, Y = np.meshgrid(X, Y)

print("X is: ")
print(X)

print("Y is: ")
print(Y)

#R = np.sqrt(X**2 + Y**2)
#surf = ax.plot_wireframe(X, Y, R, rstride=1, cstride=1)
#surf = ax.plot_wireframe(X, Y, R, rstride=1, cstride=1, cmap=cm.coolwarm, 
#    linewidth=0, antialiased=False, shade=True)
surf = ax.plot_wireframe(X, Y, soln_full, rstride=1, cstride=1)

plt.show()
raw_input("Press return to continue")
