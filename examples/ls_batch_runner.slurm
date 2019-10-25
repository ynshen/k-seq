#!/bin/bash -l
# ask for 40 cores on 1 node
#SBATCH --nodes=1 --ntasks-per-node=40
#SBATCH --time=48:00:00
#SBATCH --mail-user=yuningshen@ucsb.edu
#SBATCH --mail-type=start,end

#cd $SLURM_SUBMIT_DIR


SAMPLE_NAME='bs_pct_res_bounded'

python ls_batch_runner.py --table_path ~/byo_doped.pkl \
    --fit_partial 10 \
    --bootstrap_num 10 \
    --bs_return_num 5 \
    --bs_method 'pct_res' \
    --core_num 40 \
    --output_dir ./$SAMPLE_NAME \
    --pkg_path ../src/pkg \
    &> $SAMPLE_NAME.log

