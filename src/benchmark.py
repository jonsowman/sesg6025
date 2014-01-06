# SESG6025 Benchmarking
# Jon Sowman 2013
import multi
import timeit
import numpy
import matplotlib.pyplot as plt

class args():
    n = 3
    verbosity = 0
    its = 10000
    exercises = [1]

def run_all():
    multi.run_exercises(args.n, args.its, args.verbosity, args.exercises)

if __name__ == '__main__':
    # How many values of n (from 1, then 3, 5, 7...) should we test?
    n = 20

    results = numpy.array([])
    
    for n in range(1, n):
        t = (n * 2) + 1
        print("Benchmarking for n=%d" % t)
        args.n = t
        mtime = min(timeit.Timer(run_all).repeat(5, 1))
        print("%f" % mtime)
        results = numpy.append(results, mtime)

    print results
    plt.plot(results)
    plt.xlabel("Grid size N")
    plt.ylabel("Run time (seconds)")
    plt.title("Execution time for exercise 1")
    plt.show()
