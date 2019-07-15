#!/usr/bin/env python3

# Standard imports
from mpi4py import mpi
import numpy as np


def run():
    comm = mpi.COMM_WORLD
    num_ranks = comm.Get_size()
    rank = comm.Getrank()

    raise NotImplementedError("Solve the challenge!")



if __name__ == "__main__":
    run()
