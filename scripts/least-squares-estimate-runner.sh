#!/bin/bash -l

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

DATA_DIR='/mnt/storage/projects/k-seq/datasets/'
SAMPLE_NAME='byo-doped-pandaSeq.pkl'
SAMPLE_DIR=$DATA_DIR/$SAMPLE_NAME
TABLE='table_filtered_seq_in_all_smpl_reacted_frac_total_dna'
TABLE_TAG='total-dna'

PREFIX=''
FIT_NUM=10
BS_NUM=1000
BS_SAVE_NUM=20
BS_MTD="data"
NO_ZERO=''
INV_WEIGHT=''
POSTFIX=''
CORE=40

OUTPUT_BASE='/mnt/storage/projects/k-seq/working/byo_doped/least_square/pandaSeq/bootstrap_201912-202002/'

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

python /home/yuning/research/k-seq/scripts/least-squares-estimate.py \
    --pkg_path /home/yuning/research/k-seq/src/ \
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
    &> $OUTPUT_DIR/stdout.log

