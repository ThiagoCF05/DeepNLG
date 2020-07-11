#!/bin/sh
# Distributed under MIT license

# this sample script preprocesses a sample corpus, including tokenization,
# truecasing, and subword segmentation.
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,

script_dir=`dirname $0`
# temporary variables
. $script_dir/tmp
# variables (toolkits; source and target language)
. $script_dir/vars

main_dir=$script_dir/..
data_dir=$task_dir/data

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=20000

#minimum number of times we need to have seen a character sequence in the training text before we merge it into one unit
#this is applied to each training text independently, even with joint BPE
bpe_threshold=50

# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
$moses_scripts/training/clean-corpus-n.perl $data_dir/train $src $trg $data_dir/corpus.clean 1 80

# train truecaser
$moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/corpus.clean.$trg -model $data_dir/truecase-model.$trg

# apply truecaser (cleaned training corpus)
for prefix in corpus
 do
  cat $data_dir/$prefix.clean.$src > $data_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $data_dir/truecase-model.$trg < $data_dir/$prefix.clean.$trg > $data_dir/$prefix.tc.$trg
 done

# apply truecaser (dev/test files)
for prefix in dev test
 do
  cat $data_dir/$prefix.$src > $data_dir/$prefix.tc.$src
  $moses_scripts/recaser/truecase.perl -model $data_dir/truecase-model.$trg < $data_dir/$prefix.$trg > $data_dir/$prefix.tc.$trg
 done

# train BPE
$bpe_scripts/learn_joint_bpe_and_vocab.py -i $data_dir/corpus.tc.$trg --write-vocabulary $data_dir/vocab.$trg -s $bpe_operations -o $data_dir/$trg.bpe

# apply BPE

for prefix in corpus dev test
 do
  cat $data_dir/$prefix.tc.$src > $data_dir/$prefix.bpe.$src
  $bpe_scripts/apply_bpe.py -c $data_dir/$trg.bpe --vocabulary $data_dir/vocab.$trg --vocabulary-threshold $bpe_threshold < $data_dir/$prefix.tc.$trg > $data_dir/$prefix.bpe.$trg
 done

# build network dictionaries for separate source / target vocabularies
python3 $nematus_home/data/build_dictionary.py $data_dir/corpus.bpe.$src $data_dir/corpus.bpe.$trg

# build network dictionary for combined source + target vocabulary (for use
# with tied encoder-decoder embeddings)
cat $data_dir/corpus.bpe.$src $data_dir/corpus.bpe.$trg > $data_dir/corpus.bpe.both
python3 $nematus_home/data/build_dictionary.py $data_dir/corpus.bpe.both