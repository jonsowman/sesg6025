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
    print("[Ex1] del-squared u is 2 as required")

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

print('================')

# THIS IS THE FINAL MATRIX
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
# Could use iterative method from first part of notes
soln = numpy.linalg.solve(a, b)

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
