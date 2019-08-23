#!/usr/bin/env sh
# Distributed under MIT license

script_dir=`dirname $0`
# temporary variables
. $script_dir/tmp
# variables (toolkits; source and target language)
. $script_dir/vars

main_dir=$script_dir/../
data_dir=$task_dir/data
working_dir=$task_dir/rnn/rnn$run

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
devices=0,1,2

if [ "$task" = "lexicalization" ] || [ "$task" = "end2end" ];
then
  dataset=$data_dir/corpus.bpe
  dictionary=$data_dir/corpus.bpe.both

else
  dataset=$data_dir/corpus
  dictionary=$data_dir/corpus.both
fi

# Training command that closely follows the 'base' configuration from the
# paper
#
#  "Attention is All you Need" in Advances in Neural Information Processing
#  Systems 30, 2017. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
#  Uszkoreit, Llion Jones, Aidan N Gomez, Lukadz Kaiser, and Illia Polosukhin.
#
# Depending on the size and number of available GPUs, you may need to adjust
# the token_batch_size parameter. The command used here was tested on a
# machine with four 12 GB GPUS.
CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $dataset.$src \
    --target_dataset $dataset.$trg \
    --dictionaries $dictionary.json \
                   $dictionary.json \
    --save_freq 10000 \
    --model $working_dir/model \
    --reload latest_checkpoint \
    --model_type rnn \
    --embedding_size 300 \
    --state_size 512 \
    --tie_encoder_decoder_embeddings \
    --tie_decoder_embeddings \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 10e-09 \
    --rnn_enc_depth 1 \
    --rnn_dec_depth 1 \
    --rnn_dec_deep_context \
    --rnn_dropout_embedding 0.2 \
    --rnn_dropout_hidden 0.2 \
    --rnn_layer_normalisation \
    --rnn_dropout_source 0.1 \
    --rnn_dropout_target 0.1 \
    --maxlen 100 \
    --batch_size 80 \
    --valid_source_dataset $data_dir/dev.$eval \
    --valid_target_dataset $data_dir/references/dev.$trg"1" \
    --valid_batch_size 80 \
    --valid_freq 5000 \
    --valid_script $script_dir/validate.sh \
    --disp_freq 1000 \
    --sample_freq 9000 \
    --beam_freq 0 \
    --beam_size 5 \
    --patience 30 \
    --finish_after 200000 \
    --translation_maxlen 100