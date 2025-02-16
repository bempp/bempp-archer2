#!/bin/bash
# Simple experimental script to test bempp-rs with slurm

# We will run on a single node with 32 tasks (MPI processes) and 4 threads per process


# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=laplace_single_layer
#SBATCH --time=0:10:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=4

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=e738
#SBATCH --partition=standard
#SBATCH --qos=standard

module load PrgEnv-aocc
module load craype-network-ucx
module load cray-mpich-ucx

export WORK="/work/e738/e738/betcke"
export HOME="/home/e738/e738/betcke"

# The project directory
export PROJECT=${WORK}/projects/bempp-archer2

export SCRATCH=${PROJECT}/scratch/laplace_single_layer_${SLURM_JOBID}

# Load Spack

source ${WORK}/spack/share/spack/setup-env.sh

# Load BLAS
spack load openblas

mkdir -p ${SCRATCH}
cd ${SCRATCH}

# Propagate the cpus-per-task setting from script to srun commands
#    By default, Slurm does not propagate this setting from the sbatch
#    options to srun commands in the job script. If this is not done,
#    process/thread pinning may be incorrect leading to poor performance
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK


# Name of the script to run
script_name="laplace_single_layer_fmm_validate"

# Set the parameters for the task distribution
cores_per_node=128
tasks_per_node=5
threads_per_task=4

# We set the number of Openblas threads to 1. We don't require its threading
# and don't want it to interfere with the Rayon threads and MPI
export OPENBLAS_NUM_THREADS=1

# We also need to set the number of threads that Rayon is usiing
export RAYON_NUM_THREADS=$threads_per_task

# Now the FMM parameters

# The number of leafs in the global tree determines the global split of the
# data. The number of points is split equally across the leafs of the global tree
# A global tree depth of 2 means thefore that there are 8^2=64 leafs onto which
# the points are distributed.
global_tree_depth=4;

# The local tree depth determines how much the elements on each leaf of the
# global tree are further refined.
local_tree_depth=3;

# We now specify the output file
export OUTPUT=${SCRATCH}/laplace_single_layer_fmm_validate_output_${SLURM_JOBID}

# We now call srun to start the actual job

srun --hint=nomultithread --distribution=block:block \
"${PROJECT}/target/release/${script_name}" 


