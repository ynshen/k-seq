#!/bin/bash
# Count all fasta/fastq files in the folder and export how many reads for each file
# File naming should have the format of 'fast[a/q]' for fasta/fastq files,
#   '_L[lane]' for lane number, and `R[0/1]` for read direction
#   values in [...] will be extracted

# Usage:
#   count-fastx.sh /path/to/fastx_dir /path/to/output

# NOTE:
#   this script uses GNU grep (not the grep from Mac coreutil),
#   if you are running on MacOS, ensure the GNU grep is installed


[ -z "$1" ] && FILE_DIR=$(pwd) || FILE_DIR=$1
[ -z "$2" ] && OUTPUT_DIR=$(pwd) || OUTPUT_DIR=$2

echo "Survey files in $FILE_DIR"
OUTPUT_FILE=$OUTPUT_DIR/reads.txt

FILES=$(find "$FILE_DIR" -type f -name '*fast*' | sort)
for fastx in $FILES
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
  gz=$(echo "$fastx_name" | grep -oP 'gz')
  fasta=$(echo "$fastx_name" | grep -oP 'fasta')
  [ -z "$gz" ] && reads=$(wc -l "$fastx" | awk '{print $1}') || reads=$(gunzip -c "$fastx" | wc -l)
  echo $reads
  [ -z "$fasta" ] && reads=$((reads/4)) || reads=$((reads/2))
  echo "$fastx_name:$reads"
  if ! [ -f "$OUTPUT_FILE" ]; then
    printf "file\t%s%sreads\n" \
      "$([ -z "$lane" ] && echo "" || printf "lane\t")" \
      "$([ -z "$direction" ] && echo "" || printf "r\t")" \
      >> "$OUTPUT_FILE"
  fi
  printf "$sample\t%s%s$reads\n" \
    "$([ -z "$lane" ] && echo "" || printf "%s\t" "$lane")" \
    "$([ -z "$direction" ] && echo "" || printf "%s\t" "$direction")" \
    >> "$OUTPUT_FILE"
done

echo "Results saved to $OUTPUT_FILE"
