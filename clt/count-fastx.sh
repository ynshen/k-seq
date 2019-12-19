#!/bin/bash
# Count all fasta/fastq files in the folder and export how many reads for each file
# Naming should contains

[ -z "$1" ] && FILE_DIR=$(pwd) || FILE_DIR=$1
[ -z "$2" ] && OUTPUT_DIR=$(pwd) || OUTPUT_DIR=$2

echo "Survey files in $FILE_DIR"
OUTPUT_FILE=$OUTPUT_DIR/reads.txt

for fastx in "$FILE_DIR"/*fast*
do
  if ! [ -f "$fastx" ]
    then
      echo "No fasta/fastq file found in $FILE_DIR"
      exit 1
  fi
  fastx_name=$(basename "$fastx")
  sample=${fastx_name/_L0*}
  lane=$(echo "$fastx_name" | grep -oP '(?<=_L)\d+')
  direction=$(echo "$fastx_name" | grep -oP '(?<=_R)\d')
  gz=$(echo "$fastx_name" | grep -oP 'agz')
  fasta=$(echo "$fastx_name" | grep -oP 'fasta')
  [ -z "$gz" ] && reads=$(gunzip -c "$fastx" | wc -l) || reads=$(wc -l "$fastx")
  [ -z "$fasta" ] && reads=$((reads/4)) || reads=$((reads/2))
  echo "$fastx_name:$reads"
  if [ -f "$OUTPUT_FILE" ]; then
    printf "$sample\t%s%s$reads\n" \
      "$([ -z "$lane" ] && echo "" || printf "%s\t" "$lane")" \
      "$([ -z "$direction" ] && echo "" || printf "%s\t" "$direction")" \
      >> "$OUTPUT_FILE"
  else
    printf "file\t%s%sreads\n" \
      "$([ -z "$lane" ] && echo "" || printf "lane\t")" \
      "$([ -z "$direction" ] && echo "" || printf "r\t")" \
      >> "$OUTPUT_FILE"
  fi

done

echo "Results saved to $OUTPUT_FILE"
