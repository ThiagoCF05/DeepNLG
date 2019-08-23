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
working_dir=$task_dir/$model

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
# Currently translate.py only uses a single GPU so there is no point
# assigning more than one.
devices=0

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
     -i $pipeline_dir/$input \
     -o $pipeline_dir/$output \
     -k 5 \
     -n

# postprocess
$script_dir/postprocess.sh < $pipeline_dir/$output > $pipeline_dir/$output.postprocessed