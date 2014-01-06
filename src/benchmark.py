# SESG6025 Benchmarking
# Jon Sowman 2013
import multi
import timeit

class args():
    n = 3
    verbosity = 0
    its = 10000
    exercises = [1,2,3,4]

def run_all():
    multi.run_exercises(args.n, args.its, args.verbosity, args.exercises)

if __name__ == '__main__':
    for n in range(1, 10):
        t = (n * 2) + 1
        args.n = t
        mtime = min(timeit.Timer(run_all).repeat(1, 1))
        print("n = %d, time was %f secs" % (t, mtime))
