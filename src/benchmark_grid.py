# SESG6025 Benchmarking
# Jon Sowman 2013
import multi
import timeit
import numpy
import matplotlib.pyplot as plt

class args():
    n = 3
    verbosity = -1
    its = 10000
    exercises = [3]
    omega = 1.0

def run_all():
    multi.run_exercises(args.n, args.its, args.verbosity, args.exercises, 
            args.omega)

if __name__ == '__main__':
    # Max grid size to run
    size = 51

    gridpoints = numpy.array([])
    results = numpy.array([])
    # Run for varying grid size n
    for n in range(3, size, 2):
        print("running n = %d" % n)
        args.n = n
        repeats = 15 if n < 30 else 5
        mtime = min(timeit.Timer(run_all).repeat(repeats, 1))
        gridpoints = numpy.append(gridpoints, n)
        results = numpy.append(results, mtime)

    plt.plot(gridpoints, results)
    plt.xlabel("Grid size N")
    plt.ylabel("Run time (seconds)")
    plt.title("Execution time for exercise 3 vs grid size")
    plt.show()
