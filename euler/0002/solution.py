#!/usr/bin/env python3

# Standard imports
from mpi4py import MPI as mpi
import numpy as np
import math


def run():
    comm = mpi.COMM_WORLD
    num_ranks = comm.Get_size()
    rank = comm.Get_rank()

    # Have the root node generate the fibonacchi numbers since it's not
    # loop-independent.
    fibs = None
    if rank == 0:
        fibs = [1,2]
        while True:
            fib = sum(fibs[-2:])
            if fib > 4000000:
                break
            fibs.append(fib)

        # Master calculates the length of the input data.
        num_fibs = len(fibs)
        fibs = np.array(fibs, dtype="i")
    else:
        # All other nodes just allocate the variable.
        num_fibs = None

    # Sync the size of the incoming data to all nodes.
    num_fibs = comm.bcast(num_fibs, root=0)

    # All slave nodes allocate room for the input data.
    if rank != 0:
        fibs = np.empty(num_fibs, dtype="i")

    # Synchronize fibs to all nodes.
    comm.Bcast(fibs, root=0)

    # Prep work is now done and we can start actually doing work. So many lines
    # just to share a few numbers. :P

    # Check the data
    valid_data = []
    for fib in fibs[rank::num_ranks]:
        # Check if fib is even valued.
        if fib % 2 == 0:
            valid_data.append(fib)

    # Convert to something transferable.
    valid_data = np.array(sum(valid_data))
    print("Done: %i" % rank)

    # Gather the numbers back to root.
    gather_data = None if rank != 0 else np.zeros(num_ranks, dtype="i")
    comm.Gather(valid_data, gather_data, root=0)
    if rank == 0:
        print("SUM = %i" % sum(gather_data))

if __name__ == "__main__":
    run()
