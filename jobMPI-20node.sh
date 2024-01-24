#!/bin/bash -l
#PBS -l select=20:system=polaris
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:grand
#PBS -q prod
#PBS -A FourierHPO

set -e


cd /home/iamyixuan/work/ImPACTs/HPO/

OUTPUT_RUN="/home/iamyixuan/work/ImPACTs/HPO/output_run/"

# module load PrgEnv-gnu
# module load llvm
# module load conda/2023-10-04

source /home/iamyixuan/miniforge3/etc/profile.d/conda.sh
conda activate ml
#conda activate /home/iamyixuan/dhenv
#conda activate /home/iamyixuan/work/ImPACTs/HPO/dh


#cd ${PBS_O_WORKDIR}



export timeout=3500

# export PMI_LOCAL_RANK
# export PMI_RANK
export NGPUS_PER_NODE=4
export NDEPTH=8
export NRANKS_PER_NODE=$NGPUS_PER_NODE
export NNODES=`wc -l < $PBS_NODEFILE`
export NTOTRANKS=$(($NNODES * $NRANKS_PER_NODE))
export OMP_NUM_THREADS=$NDEPTH
export RANKS_HOSTS=$(python ./subFiles/get_hosts_polaris.py)
#!!! CONFIGURATION - END

# mkdir -p $OUTPUT_RUN
# pushd $OUTPUT_RUN


#-----------------
#pip show deephyper

/opt/cray/pe/pals/1.2.11/bin/mpiexec -n ${NTOTRANKS} --ppn ${NGPUS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth --envall /home/iamyixuan/work/ImPACTs/HPO/subFiles/set_affinity_gpu_polaris.sh  \
python hpo.py

#gzip -9 results.csv
