#!/usr/bin/env bash
# Copyright 2021  Bofeng Huang
# this is a basic script which generates lang dir


stage=0

train_set=train
dict=data/local/dict_nosp

. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 1 ]; then
  # generate lang dir
  utils/prepare_lang.sh $dict "<UNK>" data/local/lang_tmp_nosp data/lang_nosp
fi

if [ $stage -le 2 ]; then
  # generate language model
  lm_dir=data/local/lm
  mkdir -p $lm_dir
  # use train corpus to generate language model
  cp data/$train_set/oov/text_clean_lm $lm_dir

  # 3-gram
  ngram-count -order 3 -text $lm_dir/text_clean_lm -kndiscount -interpolate -lm $lm_dir/lm_tgsmall.arpa.gz
  # 4-gram
  # ngram-count -order 4 -text $lm_dir/text_clean_lm -kndiscount -interpolate -lm $lm_dir/lm_fgsmall.arpa.gz

  # evaluate ppl of test corpus on lm
  # for x in $test_sets; do
  #   [ -f data/$x/.compute_ppl.txt ] && continue

  #   echo "Evaluate PPL on data/$x"

  #   # 3-gram
  #   ngram -order 3 -lm $lm_dir/lm_tgsmall.arpa.gz -ppl <(cut -d ' ' -f 2- data/$x/text) >data/$x/.compute_ppl.txt
  #   ngram -order 3 -lm $lm_dir/lm_tgsmall.arpa.gz -ppl <(cut -d ' ' -f 2- data/$x/text) -debug 2 >data/$x/.compute_ppl_debug_3_gram.txt
  #   # 4-gram
  #   ngram -order 4 -lm $lm_dir/lm_fgsmall.arpa.gz -ppl <(cut -d ' ' -f 2- data/$x/text) >>data/$x/.compute_ppl.txt
  #   ngram -order 4 -lm $lm_dir/lm_fgsmall.arpa.gz -ppl <(cut -d ' ' -f 2- data/$x/text) -debug 2 >data/$x/.compute_ppl_debug_4_gram.txt
  # done
fi

if [ $stage -le 3 ]; then
  # prepare grammar
  # prepares the test time language model(G) transducers
  local_bh/format_lms.sh --src-dir data/lang_nosp $lm_dir
fi

# if [ $stage -le 4 ]; then
#   # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
#   utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
#   data/lang_nosp data/lang_nosp_test_tglarge
# fi
