#!/usr/bin/env python3

# Standard imports
from mpi4py import MPI as mpi
import numpy as np
import math

def is_prime(n) :
    # Corner cases.
    if (n <= 1) :
        return False
    if (n <= 3) :
        return True

    # This is checked so that we can skip middle five numbers in below loop.
    if (n % 2 == 0 or n % 3 == 0) :
        return False

    i = 5
    while(i * i <= n) :
        if (n % i == 0 or n % (i + 2) == 0) :
            return False
        i = i + 6

    return True

def run():
    comm = mpi.COMM_WORLD
    num_ranks = comm.Get_size()
    rank = comm.Get_rank()

    # Number given by challenge to prime-factorize.
    prime_in = 600851475143


    # We're not clever, so we generate a number of possible factors and just
    # straight up test them.
    # TODO: send "ranges/intervals" to the nodes instead of concrete lists.
    suspects = None
    if rank == 0:
        # Limits for computational area.
        lower_limit = 2
        upper_limit = math.ceil(math.sqrt(prime_in))

        # We don't wan't to test even numbers, or numbers that end with 5,
        # since they can't be prime numbers.
        suspects = [i for i in range(lower_limit,upper_limit) if i == 5 or (i % 2 and i % 5)]
        num_suspects = len(suspects)
        suspects = np.array(suspects, dtype="i8")
    else:
        num_suspects = None

    # Synchronize the number of testable numbers to each node.
    num_suspects = comm.bcast(num_suspects, root=0)

    # All slave nodes allocate room for the input data.
    if rank != 0:
        suspects = np.empty(num_suspects, dtype="i8")

    # Synchronize suspects to all nodes.
    comm.Bcast(suspects, root=0)

    # Take out our data, and reverse, we wan't only the largest number, so
    # starting from the top and going down makes sense.
    my_suspects = list(suspects[rank::num_ranks])
    list.reverse(my_suspects)

    # time to get to work
    largest_prime = 0
    for suspect in my_suspects:
        # If suspect is a factor and a prime, save and exit loop.
        if not prime_in % suspect and is_prime(suspect):
            largest_prime = suspect
            break

    print("Done: %i" % rank)

    # Prepare aggregation.
    largest_prime = np.array(largest_prime, dtype="i8")
    value_max = np.array(0.0, dtype="i8")
    comm.Reduce(largest_prime, value_max, op=mpi.MAX, root=0)

    # Report result.
    if rank == 0:
        print("Largest prime-factor: %i" % value_max)

if __name__ == "__main__":
    run()
