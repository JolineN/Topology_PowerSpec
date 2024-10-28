#!/bin/bash
#SBATCH -J topo              # Job name
#SBATCH -o topo.%j.out       # Name of stdout output file (%j expands to %jobId)
#SBATCH -N 1                # Total number of nodes requested
#SBATCH -n 8               # Total number of mpi tasks #requested
#SBATCH --mem=500G
#SBATCH -t 30:00:00         # Run time (hh:mm:ss) - 1.5 hour
#SBATCH --exclusive

python test_local.py
