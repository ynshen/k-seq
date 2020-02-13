#!/bin/bash -l
# ask for 40 cores on 1 node
#SBATCH --nodes=1 --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --mail-user=yuningshen@ucsb.edu
#SBATCH --mail-type=start,end

cd $SLURM_SUBMIT_DIR

#############################################  Define task ####################################################
#  Sample options:
#    - byo-doped.pkl
#    - byo-selected.pkl
#  Table name options for BYO-doped
#      - table_filtered_reacted_frac_total_dna
#      - table_filtered_reacted_frac_spike_in
#      - table_filtered_seq_in_all_smpl_reacted_frac_spike_in
#      - table_filtered_seq_in_all_smpl_reacted_frac_total_dna
#  Table name options for BYO-selected
#      - table_nf_filtered_reacted_frac_curated
#      - table_nf_filtered_seq_in_all_smpl_reacted_frac_curated
#      - table_nf_filtered_reacted_frac
#      - table_nf_filtered_seq_in_all_smpl_reacted_frac
######################################## Define estimator details #############################################
# output nameing: prefix _ table _  bs-num _ bs-mtd _ no-zero _ inv-weight _ core _ postfix
###############################################################################################################

DATA_DIR='/home/yshen/k-seq/datasets/'
SAMPLE_NAME='byo-selected.pkl'
SAMPLE_DIR=$DATA_DIR/$SAMPLE_NAME
TABLE="table_nf_filtered_reacted_frac_curated"
TABLE_TAG='byo-selected-curated'

PREFIX=''
FIT_NUM=-1
BS_NUM=0
BS_SAVE_NUM=0
BS_MTD=''
NO_ZERO=''
INV_WEIGHT=''
POSTFIX=''
CORE=40

OUTPUT_BASE='/home/yshen/k-seq/results/byo-selected'

####################################### CREATE OUTPUT DIR AND RUN ESTIMATION ##################################

FOLDER_NAME="$( [ -z $PREFIX ] && echo '' || echo ${PREFIX}_ )\
table-${TABLE_TAG}_bs-num-${BS_NUM}_bs-mtd-${BS_MTD}_\
no-zero-$( [ -z $NO_ZERO ] && echo false || echo true )_\
inv-weight-$( [ -z $INV_WEIGHT ] && echo false || echo true )_\
core-$CORE\
$( [ -z $POSTFIX ] && echo '' || echo _$POSTFIX )"

OUTPUT_DIR=$OUTPUT_BASE/$FOLDER_NAME
echo "Create folder ${OUTPUT_DIR}"
mkdir -p $OUTPUT_DIR

# Avaiable flags
# --simu_data
# --seq_table
# --table_name
# --exclude_zero
# --inverse_weight
# --stream-results
# --overwrite

python /home/yshen/k-seq/scripts/least-squares-estimate.py \
    --pkg_path /home/yshen/k-seq/src/ \
    --seq_table $SAMPLE_DIR \
    --table_name $TABLE \
    --fit_partial $FIT_NUM \
    --bootstrap_num $BS_NUM \
    --bs_record_num $BS_SAVE_NUM \
    --bs_method $BS_MTD \
    --deduplicate \
    --stream-results \
    --core_num $CORE \
    ${NO_ZERO:+--exclude_zero} \
    ${INV_WEIGHT:+--inverse_weight} \
    --output_dir $OUTPUT_DIR \

