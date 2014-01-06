# SESG6025 Benchmarking
# Jon Sowman 2013
import multi

class args():
    n = 3
    verbosity = 0
    its = 10000
    exercises = [1,2,3,4]

multi.run_exercises(args.n, args.its, args.verbosity, args.exercises)
