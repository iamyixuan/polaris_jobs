#!/bin/bash -l
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -l walltime=70:00:00
#PBS -q preemptable
#PBS -A FourierHPO

source /home/iamyixuan/miniforge3/etc/profile.d/conda.sh
conda activate ml
cd /home/iamyixuan/work/ImPACTs/HPO/
python hpo.py