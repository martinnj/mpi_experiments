# MPI Experiments

Number of experiements with multinode MPI clusters.

## How to run

- Go to the directory of the appropriate problem/case.
- Run the command `mpiexec -hostfile /path/to/inventory -N <threads-pr-host> python3 script.py`

Because I'm not too familiar with how to make sensible dynamic distribitions
of the problem spaces in MPI, the Euler solutions will only run with specific
`n` and `N` settings.

## Plans

- Project Euler things
- Scheduler that can take tasks from a SQL table/webapi and start MPI tasks.
