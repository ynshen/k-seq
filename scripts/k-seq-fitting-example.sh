#!/bin/bash -l

#############################################  Note  ####################################################
#  Table (-t) for each dataset used in the paper "Kinetic sequencing (k-Seq) as a massively
#    parallel assay for ribozyme kinetics: utility and critical parameters" listed below
#
#  byo-variant.pkl
#    - reacted_frac_qpcr: reacted fraction for all analyzable sequences, quantified using qPCR/Qubit
#    - reacted_frac_qpcr_2mutants: for only sequences within 2 Hamming distance of family center
#
#  byo-enriched.pkl
#    - nf_filtered_reacted_frac_curated: all analyzable sequences
#
#  simu-count.pkl
#    - reacted_frac: reacted fraction for analyzable sequences
#
#  Other tables can be found under `.table` property in each dataset object

######################################## Example #############################################
# here is an example to fit the first 5 sequences in the `reacted_frac_qcr` table of `byo-variant.pkl` dataset
#   with 100 bootstrap and 20 repeated fitting

least-squares-estimate.py \
    -i byo-variant.pkl \
    -t reacted_frac_qpcr \
    -o /path/to/output/folder \
    --pkg_path path/to/k-seq/src/ \
    --large_data \
    --fit_top_n 5 \
    --bootstrap_num 100 \
    --bs_record_num 20 \
    --bs_method rel_res\
    --convergence_num 20 \
    --core_num 8
