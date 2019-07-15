#!/usr/bin/env python3

# Standard imports
from mpi4py import MPI as mpi
import numpy as np
import math


def run():
    comm = mpi.COMM_WORLD
    num_ranks = comm.Get_size()
    rank = comm.Get_rank()
    problem_size = 1000

    # Check if input is partionable with our cluster size.
    if not problem_size % num_ranks == 0:
        if rank == 0:
            print("Number of ranks is not evenly divisble with problem size. Please make sure 1000 is evenly divisble with number of ranks.")
            print("Problem Size = %i" % problem_size)
            print("Cluster Size = %i" % num_ranks)
        exit(0)

    # Calculate partition size.
    data_pr_rank = int(problem_size/num_ranks)


    # If we're the master, generate workloads.
    data = None
    if rank == 0:
        data = np.array(list(range(0, problem_size)))

    # Scatter the workloads to all nodes.
    work_data = np.empty(data_pr_rank, dtype="i")
    comm.Scatter(data, work_data, root=0)

    # Check the data
    valid_data = []
    for element in work_data:
        if element % 3 == 0 or element % 5 == 0:
            valid_data.append(element)

    # Convert to something transferable.
    valid_data = np.array(valid_data)
    print("Done: %i" % rank)

    #Gather the data and calculate the sum.
    gather_data = None if rank != 0 else np.zeros(problem_size, dtype="i")
    comm.Gather(valid_data, gather_data, root=0)
    if rank == 0:
        print("SUM = %i" % sum(gather_data))

if __name__ == "__main__":
    run()
