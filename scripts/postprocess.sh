#!/bin/sh
# Distributed under MIT license

# this sample script postprocesses the MT output,
# including merging of BPE subword units,
# detruecasing, and detokenization

script_dir=`dirname $0`
# temporary variables
. $script_dir/tmp
# variables (toolkits; source and target language)
. $script_dir/vars

main_dir=$script_dir/../

if [ "$task" = "end2end" ] || [ "$task" = "end2end_augmented" ];
then
  sed -r 's/\@\@ //g' |
  $moses_scripts/recaser/detruecase.perl |
  $moses_scripts/tokenizer/normalize-punctuation.perl -l $lng |
  $moses_scripts/tokenizer/detokenizer.perl -l $lng
elif [ "$task" = "lexicalization" ];
then
  sed -r 's/\@\@ //g'
  #$moses_scripts/recaser/detruecase.perl |
  #$moses_scripts/tokenizer/detokenizer.perl -l $lng
else
  sed -r 's/\@\@ //g'
fi