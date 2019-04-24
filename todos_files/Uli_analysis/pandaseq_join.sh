#!/bin/bash

# This script join the forward and reward reads with absolute matching overlapped regions
# Adapted from Sam Vervanic and Celia Blanco's code by Yuning Shen (yuningshen@ucsb.edu)

for R1L1 in *L001_R1*
do
	mkdir CMPT
	basename=$(basename ${R1L1})
	base=${basename//_L*}
	R2L1=${R1L1//L001_R1/L001_R2}
	R1L2=${R1L1//L001_R1/L002_R1}
	R2L2=${R1L1//L001_R1/L002_R2}

	echo "Joining $base lane 1..."
	pandaseq -f $R1L1 -r $R2L1 -F\
	-p ATGTGAGACCGAGA -q GCTGGAGCTTAACT \
        -w /home/yuning/Work/ribozyme_pred/data/uli/hiSeq/$base.joined.fastq -t 0.6 -T 6 -o 100 -C completely_miss_the_point:0

