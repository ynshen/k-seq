#!/bin/bash -l
# ask for 20 cores on 1 node
#SBATCH --nodes=1 --ntasks-per-node=20
#SBATCH --time=3:00:00
#SBATCH --mail-user=yuningshen@ucsb.edu
#SBATCH --mail-type=start,end

cd $SLURM_SUBMIT_DIR

#############################################  Define task ####################################################
#  This app conduct convergence test on least-squares fitting on POD for simulated sequences
# Avaiable flags:
# --include_1250
# --pkg_path
# --output_dir
# --seq_num
# --reps
# --cores
###############################################################################################################
python /home/yshen/k-seq/tools/estimate-least-squares-convergence-test.py \
    --pkg_path /home/yshen/k-seq/src/ \
    --output_dir /home/yshen/k-seq/results/convergence_test/no_1250 \
    --seq_num 10000 \
    --cores 20