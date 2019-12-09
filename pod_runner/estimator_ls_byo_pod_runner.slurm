#!/bin/bash -l
# ask for 40 cores on 1 node
#SBATCH --nodes=1 --ntasks-per-node=30
#SBATCH --time=48:00:00
#SBATCH --mail-user=yuningshen@ucsb.edu
#SBATCH --mail-type=start,end

cd $SLURM_SUBMIT_DIR

############################  Define task sample ####################
#   table name options for BYO-doped
#      - table_filtered_reacted_frac_total_dna
#      - table_filtered_reacted_frac_spike_in
#      - table_in_all_samples 
#####################################################################

SAMPLE_NAME='byo_doped.pkl'
TABLE='total_dna'
OUTPUT_BASE='/home/yshen/k-seq/results/byo-doped_2019-10/point-est-bounded_2019-11-23'
TABLE_NAME=table_filtered_reacted_frac_$TABLE
CORE=30

############################ Define estimator details ##############################
# output nameing: prefix _ table _  bs-num _ bs-mtd _ no-zero _ inv-weight _ core _ postfix
###########################################################################

PREFIX='prefix'
FIT_NUM=-1
BS_NUM=0
BS_SAVE_NUM=0
BS_MTD="data"
NO_ZERO=''
INV_WEIGHT=''
POSTFIX=''

FOLDER_NAME= echo $( [ -z $PREFIX ] && echo '' || echo $PREFIX\_ )table-$TABLE\_bs-num-$BS_NUM\_bs-mtd-$BS_MTD\_no-zero-$( [ -z $NO_ZERO ] && echo true || echo false )_inv-weight-$( [ -z $INV_WEIGHT ] && echo true || echo false )_core-$CORE$( [ -z $POSTFIX ] && echo '' || echo _$POSTFIX )

OUTPUT_DIR=$OUTPUT_BASE/$FOLDER_NAME
mkdir -p $OUTPUT_DIR

# Avaiable flags
# --simu_data
# --seq_table
# --table_name
# --exclude_zero
# --inverse_weight

python /home/yshen/k-seq/clt/estimator_ls_byo_runner.py \
    --pkg_path /home/yshen/k-seq/src/ \
    --seq_table /home/yshen/k-seq/datasets/$SAMPLE_NAME \
    --table_name $TABLE_NAME \
    --fit_partial $FIT_NUM \
    --bootstrap_num $BS_NUM \
    --bs_record_num $BS_SAVE_NUM \
    --bs_method $BS_MTD \
    --core_num $CORE \
    ${NO_ZERO:+--exclude_zero} \
    ${INV_WEIGHT:+--inverse_weight} \
    --output_dir $OUTPUT_DIR \
    &> $OUTPUT_DIR/sys.log


