from multiprocessing import Pool

num_cores = 5  # number of cores to run parallel optimisation
start = ["original list"]
def fun(h1_bounds):
    # start = ["original list"]  # Instantiate mutable starting point within the multiprocessing-mapped function
    start.append(h1_bounds)
    return start
h1_bounds = [(1.,2.), (3.,4.), (5.,6.), (7.,8.), (9.,10.)]

if __name__ == '__main__':
    with Pool(processes=num_cores) as pool:        
        print("Begun!")
        
        for opt_result in pool.imap_unordered(fun, h1_bounds):
            print(f"Result: {opt_result}")