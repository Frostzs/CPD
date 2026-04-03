#!/usr/bin/env bash
#SBATCH --account=f202500009hpcvlabistul2x
#SBATCH --partition=normal-x86
#SBATCH --job-name=CPD_proj
#SBATCH --output=mpi_%j.out
#SBATCH --error=mpi_%j.err
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=128
#SBATCH --time=10:00
srun docs 10000-1000000-100.in
