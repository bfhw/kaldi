#!/usr/bin/env bash
# Copyright 2021  Abdel Heba
#           2021  Bofeng Huang
# Recipe for training gmm-hmm models from scratch

# set -euo pipefail
set -e

stage=0

nj=50
decode_nj=50

train_discriminative=false # by default, don't do the GMM-based discriminative
# training.

# lexicon to convert
# lexicon=data/lexicon/montreal/FR.dict
# lexicon=data/lexicon/lexicon_final_1703_u.txt
lexicon=data/lexicon/lexicon_final_u_v3.csv
# g2p model
g2p_model=data/lexicon/montreal/g2p_models/model.fst

# train set
train_set=train_hmhm190h
# test set
test_sets=test_hmhm10h

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

boost_sil_opts=(--boost-silence 1.25)

function log_stage() {
  echo
  echo "# Stage $1: $2"
  echo "# $(date)"
  echo
}

# if [ $stage -le 0 ]; then
#   log_stage 0 "prepare data"
#   # prep data
# fi

if [ $stage -le 1 ]; then
  log_stage 1 "prepare dict"

  local_bh/prepare_dict.sh --train_set $train_set --lexicon $lexicon --g2p-model $g2p_model
fi

if [ $stage -le 2 ]; then
  # echo "$0: prepare lang"
  log_stage 2 "prepare lang"

  local_bh/prepare_lang.sh --train_set $train_set
fi

if [ $stage -le 3 ]; then
  # echo "$0: extract mfcc features"
  log_stage 3 "extract mfcc features"

  mfccdir=mfcc
  for x in $train_set $test_sets; do
    if [ ! -f data/$x/feats.scp ]; then
      steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
        data/$x exp/make_mfcc/$x $mfccdir || exit 1
      steps/compute_cmvn_stats.sh \
        data/$x exp/make_mfcc/$x $mfccdir || exit 1
      utils/fix_data_dir.sh data/$x
    else
      # but can be different
      # attention here
      echo "MFCC features of $x already exist"
    fi
  done
fi

if [ $stage -le 4 ]; then
  # echo "$0: make small data subsets"
  log_stage 4 "make small data subsets"

  # Make some small data subsets for early system-build stages
  # For the monophone stages we select the shortest utterances, which should make it easier to align the data from a flat start.
  # we want to start the monophone training on relatively short utterances
  # (easier to align), but not only the shortest ones (mostly uh-huh).
  # So take the 100k shortest ones, and
  # then take 30k random utterances from those (about 12hr)

  # utils/subset_data_dir.sh --shortest data/$train_set 100000 data/${train_set}_100k_short
  # utils/subset_data_dir.sh data/${train_set}_100k_short 30000 data/${train_set}_30k_short

  # set up for corpus hm hm
  # mono corpus
  # ~59.09h
  utils/subset_data_dir.sh --shortest data/$train_set 150000 data/${train_set}_150k_short
  # ~19.75h
  utils/subset_data_dir.sh data/${train_set}_150k_short 50000 data/${train_set}_50k_short

  # tri corpus
  # Take the first 100k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first data/$train_set 100000 data/${train_set}_100k
  # ~94.25h
  utils/data/remove_dup_utts.sh 200 data/${train_set}_100k data/${train_set}_100k_nodup

  # the full training set
  # ~184.79h
  utils/data/remove_dup_utts.sh 300 data/$train_set data/${train_set}_nodup
fi

if [ $stage -le 5 ]; then
  # echo "$0: train monophone system"
  log_stage 5 "train monophone system"

  # Possibly during training you did not put silence in your transcripts in all the places where silence actually appears (including at the beginning and end of utterances). This would force the non-silence phones to learn to model silence also.
  # It might be better to train the system using the optional silence, and then prepare a different lang directory where you disable optional silence, and use that one to do the final alignment; just make sure the phones.txt is the same.
  # Sometimes training with e.g. --boost-silence 1.25 can help to avoid the non-silence phones modeling silence
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/${train_set}_50k_short data/lang_nosp exp/mono || exit 1

  # generally train mono on short data
  # but our corpus is biased, the shortests are all 'oui' ou 'non'# steps/train_mono.sh ${boost_sil_opts[@]} --nj $nj --cmd "$train_cmd" \
  #    data/$train_set data/lang_nosp exp/mono || exit 1;

  # (
  #   # lang test (G.fst)
  #   utils/mkgraph.sh data/lang_nosp_test_tgsmall \
  #     exp/mono exp/mono/graph_nosp_tgsmall

  #   for x in $test_sets; do
  #     steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
  #       exp/mono/graph_nosp_tgsmall \
  #       data/$x \
  #       exp/mono/decode_nosp_tgsmall_${x}

  #     # steps/score_kaldi.sh --cmd "$decode_cmd" \
  #     #   data/$x data/lang_nosp exp/mono/decode_nosp_tgsmall_${x}
  #   done
  # ) &
  # wait
fi

if [ $stage -le 6 ]; then
  # echo "$0: train delta + delta-delta triphone system"
  log_stage 6 "train triphone system (1)"

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/${train_set}_100k_nodup data/lang_nosp exp/mono exp/mono_ali

  # train a first delta + delta-delta triphone system
  # <num-leaves> <tot-gauss>
  # ? swbd: 3200 30000
  #     3000 40000 \
  steps/train_deltas.sh --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    3200 30000 \
    data/${train_set}_100k_nodup data/lang_nosp exp/mono_ali exp/tri1

  (
    # decode using the tri1 model
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri1 exp/tri1/graph_nosp_tgsmall

    for x in $test_sets; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri1/graph_nosp_tgsmall \
        data/$x \
        exp/tri1/decode_nosp_tgsmall_${x}

      # ? steps/decode_si.sh

      # steps/lmrescore.sh --cmd "$decode_cmd" \
      #   $train_set/lang_nosp_test_{tgsmall,fgsmall} \
      #   $x \
      #   exp/tri1/decode_nosp_{tgsmall,fgsmall}_${test}

      #steps/lmrescore_const_arpa.sh \
      #  --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      #  $test_sets exp/tri1/decode_nosp_{tgsmall,tglarge}_${test}

      # steps/score_kaldi.sh --cmd "$decode_cmd" \
      #   data/$x data/lang_nosp exp/tri1/decode_nosp_tgsmall_${x}
    done
  ) &
  wait
fi

if [ $stage -le 7 ]; then
  # echo "$0: train delta + delta-delta triphone system"
  log_stage 7 "train triphone system (2)"

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/${train_set}_100k_nodup data/lang_nosp exp/tri1 exp/tri1_ali

  # swbd recipe: 4000 70000
  steps/train_deltas.sh --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    4000 70000 \
    data/${train_set}_100k_nodup data/lang_nosp exp/tri1_ali exp/tri1b

  (
    # decode using the tri1b model
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri1b exp/tri1b/graph_nosp_tgsmall

    for x in $test_sets; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri1b/graph_nosp_tgsmall \
        data/$x \
        exp/tri1b/decode_nosp_tgsmall_${x}

      # ? steps/decode_si.sh

      # steps/lmrescore.sh --cmd "$decode_cmd" \
      #   $train_set/lang_nosp_test_{tgsmall,fgsmall} \
      #   $x \
      #   exp/tri1b/decode_nosp_{tgsmall,fgsmall}_${test}

      #steps/lmrescore_const_arpa.sh \
      #  --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      #  $test_sets exp/tri1b/decode_nosp_{tgsmall,tglarge}_${test}

      # steps/score_kaldi.sh --cmd "$decode_cmd" \
      #   data/$x data/lang_nosp exp/tri1b/decode_nosp_tgsmall_${x}
    done
  ) &
  wait
fi

if [ $stage -le 8 ]; then
  # echo "$0: train LDA+MLLT system"
  log_stage 8 "train LDA+MLLT system"

  # begin to use whole training set
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/${train_set}_nodup data/lang_nosp exp/tri1b exp/tri1b_ali

  # train an LDA+MLLT system.
  # 7000 100000 \
  # swbd: 6000 140000 \
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    --splice-opts "--left-context=3 --right-context=3" \
    6000 140000 \
    data/${train_set}_nodup data/lang_nosp exp/tri1b_ali exp/tri2

  # steps/train_lda_mllt.sh --cmd "$train_cmd" \
  #   --splice-opts "--left-context=3 --right-context=3" 4000 60000 \
  #   data//${train_set}_nodup data/lang_nosp exp/tri1b_ali exp/tri2

  (
    # decode using the LDA+MLLT model
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
      exp/tri2 exp/tri2/graph_nosp_tgsmall

    for x in $test_sets; do
      steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        exp/tri2/graph_nosp_tgsmall \
        data/$x \
        exp/tri2/decode_nosp_tgsmall_${x}

      #steps/lmrescore.sh --cmd "$decode_cmd" \
      #  ${train_set}_nodup/lang_nosp_test_{tgsmall,fgsmall} \
      #  $x \
      #  exp/tri2/decode_nosp_{tgsmall,fgsmall}_${test}

      #steps/lmrescore_const_arpa.sh \
      #  --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      #  data/$test_sets exp/tri2/decode_nosp_{tgsmall,tglarge}_${test}

      # steps/score_kaldi.sh --cmd "$decode_cmd" \
      #   data/$x data/lang_nosp exp/tri2/decode_nosp_tgsmall_${x}
    done
  ) &
  wait
fi

if [ $stage -le 9 ]; then
  log_stage 9 "recreate lang"
  # Now we compute the pronunciation and silence probabilities from training data, and re-create the lang directory.

  steps/get_prons.sh --cmd "$train_cmd" \
    data/${train_set}_nodup data/lang_nosp exp/tri2

  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt \
    data/local/dict

  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang_tmp data/lang

  # arpa2fst
  # data/lang_test_tgsmall
  local_bh/format_lms.sh --src-dir data/lang data/local/lm

  # carpa
  # utils/build_const_arpa_lm.sh \
  #   data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  # utils/build_const_arpa_lm.sh \
  #   data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

  (
    utils/mkgraph.sh data/lang_test_tgsmall \
      exp/tri2 exp/tri2/graph_tgsmall

    for x in $test_sets; do
      steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
        exp/tri2/graph_tgsmall \
        data/$x \
        exp/tri2/decode_tgsmall_${x}
    done
  ) &
  wait
fi

if [ $stage -le 10 ]; then
  # echo "$0: train LDA+MLLT+SAT system"
  log_stage 10 "train LDA+MLLT+SAT system"

  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/${train_set}_nodup data/lang exp/tri2 exp/tri2_ali

  # ? faster version: steps/train_quick.sh
  # swbd: 11500 200000 \
  steps/train_sat.sh --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    15000 300000 \
    data/${train_set}_nodup data/lang exp/tri2_ali exp/tri3

  (
    # decode using the tri3 model
    utils/mkgraph.sh data/lang_test_tgsmall \
      exp/tri3 exp/tri3/graph_tgsmall

    for x in $test_sets; do
      steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
        --config conf/decode.config \
        exp/tri3/graph_tgsmall \
        data/$x \
        exp/tri3/decode_tgsmall_${x}

      # steps/lmrescore.sh --cmd "$decode_cmd" \
      #   data/lang_nosp_test_{tgsmall,fgsmall} \
      #   data/$x \
      #   exp/tri3/decode_nosp_{tgsmall,fgsmall}_${test}

      #steps/lmrescore_const_arpa.sh \
      #  --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      #  data/$test_sets exp/tri3/decode_nosp_{tgsmall,tglarge}_${test}

      # steps/score_kaldi.sh --cmd "$decode_cmd" \
      #   data/$x $lang exp/tri3/decode_tgsmall_${x}

      ./local_bh/score_ali.sh data/$x exp/tri3/decode_tgsmall_${x}
    done
  ) &
  wait

  # try with another lang
  # lang_test=data/lang_nosp_test_tgsmall_merged_lv_hmhm
  # graph_dir=exp/tri3/graph_tgsmall_merged_lv_hmhm
  # decode_dir=exp/tri3/decode_tgsmall_merged_lv_hmhm
  # (
  #   # decode using the tri3 model
  #   utils/mkgraph.sh $lang_test \
  #     exp/tri3 $graph_dir

  #   for x in $test_sets; do
  #     steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" \
  #       --config conf/decode.config \
  #       $graph_dir \
  #       data/$x \
  #       ${decode_dir}_${x}

  #     # steps/lmrescore.sh --cmd "$decode_cmd" \
  #     #   data/lang_nosp_test_{tgsmall,fgsmall} \
  #     #   data/$x \
  #     #   exp/tri3/decode_nosp_{tgsmall,fgsmall}_${test}

  #     #steps/lmrescore_const_arpa.sh \
  #     #  --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
  #     #  data/$test_sets exp/tri3/decode_nosp_{tgsmall,tglarge}_${test}

  #     # steps/score_kaldi.sh --cmd "$decode_cmd" \
  #     #   data/$x $lang exp/tri3/decode_tgsmall_${x}
  #   done
  # ) &
  # wait
fi

if ! $train_discriminative; then
  echo "$0: exiting early since --train-discriminative is false."
  exit 0
fi

if [ $stage -le 11 ]; then
  log_stage 11 "train MMI system"

  # MMI training starting from the LDA+MLLT+SAT systems on all the (nodup) data.
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/${train_set}_nodup data/lang exp/tri3 exp/tri3_ali

  steps/make_denlats.sh --nj $nj --cmd "$decode_cmd" \
    --config conf/decode.config --transform-dir exp/tri3_ali \
    data/${train_set}_nodup data/lang exp/tri3 exp/tri3_denlats

  # 4 iterations of MMI seems to work well overall. The number of iterations is
  # used as an explicit argument even though train_mmi.sh will use 4 iterations by
  # default.
  num_mmi_iters=4
  steps/train_mmi.sh --cmd "$decode_cmd" \
    --boost 0.1 --num-iters $num_mmi_iters \
    data/${train_set}_nodup data/lang exp/tri3_{ali,denlats} exp/tri3_mmi_b0.1

  for x in $test_sets; do
    for iter in 1 2 3 4; do
      (
        steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --config conf/decode.config --iter $iter \
          --transform-dir exp/tri3/decode_tgsmall_${x} \
          exp/tri3/graph_tgsmall \
          data/$x \
          exp/tri3_mmi_b0.1/decode_tgsmall_${x}_${iter}
        # if $has_fisher; then
        #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        #     data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        #     exp/tri3_mmi_b0.1/decode_eval2000_${iter}.mdl_sw1_{tg,fsh_fg}
        # fi
      ) &
    done
  done
  wait
fi

if [ $stage -le 12 ]; then
  log_stage 12 "train fMMI+MMI system"

  # Now do fMMI+MMI training
  steps/train_diag_ubm.sh --silence-weight 0.5 --nj $nj --cmd "$train_cmd" \
    700 data/${train_set}_nodup data/lang exp/tri3_ali exp/tri3_dubm

  steps/train_mmi_fmmi.sh --learning-rate 0.005 \
    --boost 0.1 --cmd "$train_cmd" \
    data/${train_set}_nodup data/lang exp/tri3_ali exp/tri3_dubm \
    exp/tri3_denlats exp/tri3_fmmi_b0.1

  for x in $test_sets; do
    for iter in 4 5 6 7 8; do
      (
        steps/decode_fmmi.sh --nj $decode_nj --cmd "$decode_cmd" \
          --config conf/decode.config --iter $iter \
          --transform-dir exp/tri3/decode_tgsmall_${x} \
          exp/tri3/graph_tgsmall \
          data/$x \
          exp/tri3_fmmi_b0.1/decode_tgsmall_${x}_${iter}

        # if $has_fisher; then
        #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        #     data/lang_sw1_{tg,fsh_fg} data/eval2000 \
        #     exp/tri3_fmmi_b0.1/decode_eval2000_it${iter}_sw1_{tg,fsh_fg}
        # fi
      ) &
    done
  done
  wait
fi
