#!/bin/sh
# Distributed under MIT license

# this script evaluates the best model (according to BLEU early stopping)
# on newstest2017, using detokenized BLEU (equivalent to evaluation with
# mteval-v13a.pl)

script_dir=`dirname $0`
# temporary variables
. $script_dir/tmp
# variables (toolkits; source and target language)
. $script_dir/vars

main_dir=$script_dir/../
data_dir=$task_dir/data
working_dir=$task_dir/$model

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
# Currently translate.py only uses a single GPU so there is no point
# assigning more than one.
devices=0

#test_prefix=dev
test=$test_prefix.$eval
ref=$data_dir/references/$test_prefix.$trg
refs=$ref"1 "$ref"2 "$ref"3 "$ref"4 "$ref"5"
# ensemble the best models of the three runs
if [ "$model" = "rnn" ];
then
  model_path=$working_dir/$model"1"/model.best-valid-script" "$working_dir/$model"2"/model.best-valid-script" "$working_dir/$model"3"/model.best-valid-script
else
  model_path=$working_dir/$model"1"/model.best-valid-script
fi

# decode
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/translate.py \
     -m $model_path \
     -i $data_dir/$test \
     -o $working_dir/$test_prefix.out \
     -k 5 \
     -n

# postprocess
$script_dir/postprocess.sh < $working_dir/$test_prefix.out > $working_dir/$test_prefix.out.postprocessed

# evaluate with detokenized BLEU (same as mteval-v13a.pl)
if [ "$task" = "lexicalization" ] || [ "$task" = "end2end" ] || [ "$task" = "end2end_augmented" ];
then
  $nematus_home/data/multi-bleu-detok.perl $refs < $working_dir/$test_prefix.out.postprocessed
else
  python3 $script_dir/accuracy.py $ref $working_dir/$test_prefix.out.postprocessed
fi
