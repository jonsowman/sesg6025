#######################################
#
# SESG6025 PDE Courswork
#
# Jon Sowman 2013
# <js39g13@soton.ac.uk>
#
#######################################

import numpy
import scipy
import argparse

def verify(s, h, exno, complex=False):
    """
    Verify that the solution matrix 's' meets the requirement that the
    Laplacian is equal to 2 at coordinates (0.5, 0.5) with discretisation
    distance 'h'. Additionally takes the exercise number for printing reasons,
    and whether we're using the complex stencil or otherwise.
    """
    tol = 1e-08
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
        if c-2 in range(size):
            x = x - s[c-2,c]
            y = y - s[c,c-2]
        if c+2 in range(size):
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
        print('[ERROR Ex' + str(exno) + '] Solution verification failed: ' \
                + ' %.3f', result)
    else:
        print('[Ex' + str(exno) + '] ' + u'\u2207\u00b2' + \
                'u = 2 within ' + u'\u00b1' + str(tol) + \
                ' at u = (0.5,0.5) as required')

def sor(A, b, its, x=None, omega=1.0):
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

    # Hold the old solution vector for convergence detection
    x_old = numpy.empty(x.size)

    # Iterate 'iterations' times maximum, stop if we reach convergence
    for it in range(its):
        for i in range(n):
            t1 = 0
            t2 = 0
            for j in range(i):
                t1 = t1 + A[i, j] * x[j]
            for j in range(i+1, n):
                t2 = t2 + A[i, j] * x[j]
            x[i] = omega/A[i, i] * (b[i] - t1 - t2) + (1.0-omega)*x[i]
        if numpy.allclose(x, x_old, rtol=1e-09):
            break
        x_old = x.copy()

    # Check that we didn't hit max_its -- warn if so
    if it == (its - 1):
        print("[WARN] Iterations limit exceeded, exiting")

    return [x, it]

def embed(a):
    """
    Embed solution matrix into a matrix with the boundary conditions.
    """
    size = a.shape[0]
    a_tmp = numpy.zeros([size+2, size+2])

    for i in range(1, size+1):
        for j in range(1, size+1):
            a_tmp[i, j] = a[i-1, j-1]

    return a_tmp

def rbidx(i, j, n):
    """
    Calculate the index for a red-black node at grid coordinates (i, j) 
    given that top left node is node 0 and is always red.
    """
    idx = (i*n+j+1)/2 + (n**2)/2 if (i+j)%2 else (i*n+j)/2
    return idx

def rbstencil(n):
    """
    Construct a stencil for the red-black ordering of the adjacency matrix
    of size n^2. This Red-Black ordering is used instead of the natural
    ordering such that parallelism may be exploited.
    The top left node is always red in this implementation.
    """
    # Construct the diagonal which is -4 on each element
    A = -4 * numpy.eye(n**2)

    for i in range(n):
        for j in range(n):
            # Find the index of this node
            idx = rbidx(i, j, n)
            # If this node is red, neighbours are black & vice versa
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

def redblack(A, b, its, x=None):
    """
    Run a full red-black solution using the Gauss-Seidel method, i.e.
    solving black using the previously computed red solution vector.
    Re-order the solution vector to map it back onto the natural ordered
    grid instead of the Red-Black grid.
    """
    if x is None:
        x = numpy.zeros(len(A[0]))

    # Hold the old answer for convergence detection
    x_old = x

    # Iterate until convergence or we hit the max iterations value
    for i in range(its):
        x = rbstep(A, b, x)
        if numpy.allclose(x, x_old, rtol=1e-09):
            break
        x_old = x.copy()

    # Check that we didn't hit max_its -- warn if so
    if i == (its - 1):
        print("[WARN] Iterations limit exceeded, exiting")

    # Reorder the solution vector by interleaving the red and black
    # solution vectors
    c = x.size/2
    sol = numpy.empty(x.size, dtype=x.dtype)
    sol[0::2] = x[0:c+1]
    sol[1::2] = x[c+1:len(x)]
    return [sol, i]

def rbstep(A, b, x):
    """
    Run one iteration of the Red-Black Gauss-Seidel solver. This is
    not run in a parallel manner but could be if required.
    Note that the solution is completely parallel within the red
    solving, and ditto for the black (Db & Dr are diagonal). 
    This linear algebra formulation is based on the method at:
    http://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf
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
    xr = numpy.linalg.solve(Dr, br - F.dot(xb))

    # Solve for black using the red x-values we just calculated
    xb = numpy.linalg.solve(Db, bb - E.dot(xr))

    # Concatenate the solution vectors xr and xb and return
    return numpy.concatenate([xr, xb])

def complex_stencil(n):
    """
    Use the more complex 8 point stencil described in the coursework handout
    to construct the matrix a (correct to 4th order)
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

def simple_stencil(n, verbosity):
    """
    Use the stencil derived in lectures, the first central difference
    approximation (correct to 2nd order)
    """

    # Clear matrix and set it up
    a = numpy.zeros([n**2, n**2])

    if(verbosity >= 2):
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

    if verbosity >= 2:
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
    if verbosity >= 2:
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
    if verbosity >= 2:
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
    if verbosity >= 2:
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

    if verbosity >= 2:
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

def solve_ex1(n, h, its, verbosity):
    """
    Solve and return exercise 1, which is the simple stencil using the numpy QR
    solver
    """
    # Compute the matrices for simple and complex stencils
    if verbosity >= 1:
        print("Setting up simple stencil"),
    a_simple = simple_stencil(n, verbosity)
    if verbosity >= 1:
        print("...done")
        if verbosity >= 2:
            print("Simple stencil is")
            print(a_simple)
    
    # For rho(0.5, 0.5) = 2 we just require that the middle element of b
    # is -2*(h**2), as per equation 4.51 for the simple stencil.
    # For the complex stencil there's a factor of 12 in there too
    # For the red-black solver, the linear system of equations is in a different
    # order
    b_simple = numpy.zeros([n**2, 1])
    b_simple[((n**2)-1)/2] = 2*(h**2)
    
    if verbosity >= 2:
        print("b_simple is:")
        print(b_simple)

    # For ex1 we use the np method, for ex2 we use our own SOR method
    if verbosity >= 1:
        print("Solving with numpy QR solver"),
    ex1_soln = numpy.linalg.solve(a_simple, b_simple)
    if verbosity >= 1:
        print("...done")
        if verbosity >= 2:
            print("Solution is:")
            print(ex1_soln)

    return ex1_soln

def solve_ex2(n, h, its, verbosity):
    """
    Solve exercise 2, which is the simple stencil but using our own
    successive-over-relaxation implementation of the Gauss-Seidel iterative
    method
    """
    # Compute the matrices for simple and complex stencils
    if verbosity >= 1:
        print("Setting up simple stencil"),
    a_simple = simple_stencil(n, verbosity)
    if verbosity >= 1:
        print("...done")
        if verbosity >= 2:
            print("Simple stencil is")
            print(a_simple)
    
    # For rho(0.5, 0.5) = 2 we just require that the middle element of b
    # is -2*(h**2), as per equation 4.51 for the simple stencil.
    # For the complex stencil there's a factor of 12 in there too
    # For the red-black solver, the linear system of equations is in a different
    # order
    b_simple = numpy.zeros([n**2, 1])
    b_simple[((n**2)-1)/2] = 2*(h**2)
    
    if verbosity >= 2:
        print("b_simple is:")
        print(b_simple)

    # Solve using the sor() method
    if verbosity >= 1:
        print("Solving with Gauss-Seidel"),
    [ex2_soln, ex2_its] = sor(a_simple, b_simple, its)
    if verbosity >= 1:
        print("...done in %d iterations" % ex2_its)
        if verbosity >= 2:
            print("Solution is:")
            print(ex2_soln)

    return ex2_soln

def solve_ex3(n, h, its, verbosity):
    """
    Solve exercise 3, which is the same PDE but using a more complex stencil
    given in the lecture notes which is correct to fourth order.
    """
    if verbosity >= 1:
        print("Setting up complex stencil"),
    a_complex = complex_stencil(n)
    if verbosity >= 1:
        print("...done")
        if verbosity >= 2:
            print("Complex stencil is")
            print(a_complex)

    # For rho(0.5, 0.5) = 2 we just require that the middle element of b
    # is -2*(h**2), as per equation 4.51 for the simple stencil.
    # For the complex stencil there's a factor of 12 in there too
    # For the red-black solver, the linear system of equations is in a different
    # order
    b_complex = numpy.zeros([n**2, 1])
    b_complex[((n**2)-1)/2] = 12*2*(h**2)

    if verbosity >= 2:
        print("b_complex is:")
        print(b_complex)

    # Now let's solve the complex stencil problem
    if verbosity >= 1:
        print("Solving complex stencil using numpy"),
    ex3_soln = numpy.linalg.solve(a_complex, b_complex)
    if verbosity >= 1:
        print("...done")
        if verbosity >= 2:
            print("Solution is:")
            print(ex3_soln)

    return ex3_soln

def solve_ex4(n, h, its, verbosity):
    """
    Solve exercise 4, which uses the numpy QR solver but in a parallelisable
    ordering of the solution matrix grid points known as red-black
    """
    if verbosity >= 1:
        print("Setting up red-black stencil"),
    a_redblack = rbstencil(n)
    if verbosity >= 1:
        print("...done")
        if verbosity >= 2:
            print("Red-black stencil is")
            print(a_redblack)

    # For rho(0.5, 0.5) = 2 we just require that the middle element of b
    # is -2*(h**2), as per equation 4.51 for the simple stencil.
    # For the complex stencil there's a factor of 12 in there too
    # For the red-black solver, the linear system of equations is in a different
    # order
    b_redblack = numpy.zeros([n**2])
    b_redblack[(n**2)/4] = 2*(h**2)

    if verbosity >= 2:
        print("b_redblack is:")
        print(b_redblack)

    # Finally, solve the red-black problem
    if verbosity >= 1:
        print("Solving simple stencil in Red-Black formulation"),
    [ex4_soln, ex4_its] = redblack(a_redblack, b_redblack, its)
    if verbosity >= 1:
        print("...done in %d iterations" % ex4_its)
        if verbosity >= 2:
            print("Solution is:")
            print(ex4_soln)

    return ex4_soln

def run_exercises(n, its, verbosity):
    # The h value is 1/(n+2) : taking into account the intervals
    # to get to the boundary
    h = 1.0 / (n+2)

    # Reset printing options
    numpy.set_printoptions()
    numpy.set_printoptions(edgeitems=3, infstr='inf', linewidth=75, nanstr='nan',
            precision=8, suppress=False, threshold=1000)

    # Run the solutions, embed them in a solution matrix with boundary
    # conditions and then verify each solution in turn
    if verbosity >= 1:
        print("Exercise 1: Solve PDE using numpy QR solver and simple stencil")
    ex1_soln = solve_ex1(n, h, its, verbosity)
    ex1_full = embed(numpy.reshape(ex1_soln, [n, n]))
    verify(ex1_full, h, 1)

    if verbosity >= 1:
        print("Exercise 2: Solve PDE using SOR solver and simple stencil")
    ex2_soln = solve_ex2(n, h, its, verbosity)
    ex2_full = embed(numpy.reshape(ex2_soln, [n, n]))
    verify(ex2_full, h, 2)

    if verbosity >= 1:
        print("Exercise 3: Solve PDE using numpy QR solver and complex stencil")
    ex3_soln = solve_ex3(n, h, its, verbosity)
    ex3_full = embed(numpy.reshape(ex3_soln, [n, n]))
    verify(ex3_full, h, 3, complex=True)

    if verbosity >= 1:
        print("Exercise 4: Solve PDE using Gauss-Seidel Red/Black solver " + \
                "and simple stencil")
    ex4_soln = solve_ex4(n, h, its, verbosity)
    ex4_full = embed(numpy.reshape(ex4_soln, [n, n]))
    verify(ex4_full, h, 4)

    return [ex1_full, ex2_full, ex3_full, ex4_full]

def plot(solution, verbosity):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    steps = 2.0 + n

    h = 1.0 / (steps - 1)
    if verbosity >= 2:
        print('h = ', h)
    X = numpy.arange(0, steps, 1) * h
    if verbosity >= 2:
        print(X)
        print
    Y = numpy.arange(0, steps, 1) * h
    X, Y = numpy.meshgrid(X, Y)

    surf = ax.plot_wireframe(X, Y, solution, rstride=1, cstride=1)
    plt.show()

if __name__ == '__main__':
    # Parse the command line options
    parser = argparse.ArgumentParser(description="SESG6025 Coursework by \
            Jon Sowman. PDE solver using various methods.\n\nWe attempt to \
            solve the equation " + u'\u2207\u00b2' + "u = " + u'\u03c1' + ", \
            using the \
            following methods: [Ex1] Numpy's QR solver and a central \
            difference approximation to the second derivative correct \
            to second order. [Ex2] A Gauss-Seidel iterative method \
            implemented in this program. [Ex3] Numpy's QR solver, however \
            this time using a central difference approximation to the second \
            derivative correct to fourth order. [Ex4] A modification of \
            the Gauss-Seidel method, changing from the natural ordering of \
            grid points to a Red-Black ordering, displaying the potential \
            for exploiting parallelising the solving operation.")
    parser.add_argument("-p", "--plot", type=int, \
            help="Plot the solution to given exercise")
    parser.add_argument("-n", type=int, help="Size of the grid (nxn), \
            defaults to 3")
    parser.add_argument("-v", "--verbosity", action='count', help="Verbosity level \
            0, 1 or 2 (use -v or -vv)")
    parser.add_argument("-i", "--iterations", type=int, help="For Gauss \
            Seidel and Red-Black solvers, the maximum number of iterations \
            (defaults to 10000)")
    args = parser.parse_args()

    # Choose a verbosity level
    verbosity = args.verbosity if args.verbosity else 0

    if verbosity >= 1:
        print("=== SESG6025 Coursework ===\n===  Jon Sowman (2013)  ===")
        print("===========================")

    # Set up printing of the array so it displays nicely
    numpy.set_printoptions(precision=0, linewidth=120)

    # Choose and set n to a default value to begin with
    n_default = 3
    n = n_default

    # Determine the iterations limit for the GS methods
    its = args.iterations if args.iterations else 10000

    # If user supplied a custom value with -n, sanity check it before
    # assignment to n
    if args.n:
        if args.n < 3:
            print("[WARN] n must be at least 3, supplied %d, defaulting to n=%d" % 
                    (args.n, n_default))
        elif (args.n % 2) == 0:
            print("[WARN] n must be an odd number, supplied %d, defaulting to n=%d" %
                    (args.n, n_default))
        else:
            n = args.n

    # Now run the exercises using an NxN grid
    solns = run_exercises(n, its, verbosity)

    # Plot the solution if required
    if args.plot:
        ex = args.plot
        if verbosity >= 1:
            print("Plotting solution to ex %d" % ex)
        plot(solns[ex-1], verbosity)
