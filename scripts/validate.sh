#!/bin/sh
# Distributed under MIT license

# this script evaluates translations of the newstest2013 test set
# using detokenized BLEU (equivalent to evaluation with mteval-v13a.pl).

translations=$1

script_dir=`dirname $0`
# temporary variables
. $script_dir/tmp
# variables (toolkits; source and target language)
. $script_dir/vars

main_dir=$script_dir/../
data_dir=$task_dir/data/references

#language-independent variables (toolkit locations)
#. $main_dir/../vars

dev_prefix=dev
ref=$data_dir/$dev_prefix.$trg
refs=$ref"1 "$ref"2 "$ref"3 "$ref"4 "$ref"5"

# evaluate translations and write BLEU score to standard output (for
# use by nmt.py)
if [ "$task" = "lexicalization" ] || [ "$task" = "end2end" ];
then
  $script_dir/postprocess.sh < $translations | \
      $nematus_home/data/multi-bleu-detok.perl $refs | \
      cut -f 3 -d ' ' | \
      cut -f 1 -d ','
else
  python3 $script_dir/accuracy.py $ref $translations
fi