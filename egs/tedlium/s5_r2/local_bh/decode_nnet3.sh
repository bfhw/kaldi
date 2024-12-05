#!/usr/bin/env bash

# Copyright 2021  Bofeng Huang

# cp -r /home/bhuang/asr/models/internal/nnet3_rvb exp/ivector_mdl
# cp -r /home/bhuang/kaldi/egs/swbd/s5c/conf .

echo "$0 $@" # Print the command line for logging

stage=0

nj=64
decode_nj=50

test_sets="test_hm_hm_10h"
# test_sets="test_carglass_43h"
# test_sets="test_hm_hm_10h test_carglass_5h test_carglass_7h test_bot_hm_clean test_bot_hm_dirty"
# test_sets="test_bot_hm_clean test_bot_hm_dirty"
# test_sets="test_carglass_5h test_carglass_7h"
# test_sets="test_carglass_5h"

# ivector_dir=exp/nnet3_sp_aug_best
# ivector_dir=exp/nnet3_sp
ivector_dir=exp/nnet3

# dir=exp/chain/tdnn7r_sp_aug_best
dir=exp/chain/tdnn7r_sp_aug
# dir=exp/chain/tdnn7r_sp

# src_dir=/home/bhuang/kaldi/egs/swbd/s5c/exp/chain/tdnn7r_sp_scratch_noaug
# am=${src_dir}/final.mdl

lang_test=data/lang_nosp_test_tgsmall
# lang_test=data/lang_nosp_test_tgsmall_merged_lv_hmhm
# lang_test=data/lang_nosp_test_tgsmall_merged_lv_hmhm_carglass_manuel
# lang_test=data/lang_nosp_test_tgsmall_merged_lv_callbots

stats_wer=false
generate_ctm=false

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# todo: rm "test" in lang
lang_tag=$(basename $lang_test | sed "s/lang//" | sed "s/_test//")

graph_dir=$dir/graph$lang_tag
# graph_dir=$dir/graph${lang_tag}_1500iter

if [ $stage -le 0 ]; then
    echo "$0: extract mfcc features"
    mfccdir=mfcc_hires
    for dataset in $test_sets; do
        utils/copy_data_dir.sh data/$dataset data/${dataset}_hires

        steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
            --mfcc-config conf/mfcc_hires.conf \
            data/${dataset}_hires exp/make_mfcc/$dataset $mfccdir || exit 1;

        steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_mfcc/$dataset $mfccdir

        utils/fix_data_dir.sh data/${dataset}_hires || exit 1;
    done
fi

if [ $stage -le 1 ]; then
    echo "$0: extract ivectors"
    for dataset in $test_sets; do
        # * nj < num of speakers
        steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
            --nj $nj data/${dataset}_hires ${ivector_dir}/extractor ${ivector_dir}/ivectors_$dataset || exit 1;
    done
fi

# if [ $stage -le 2 ]; then
#     echo "$0: graph"
#     # 3 -> shift -1, 0, 1
#     # [ -f exp/frame_subsampling_factor ] || echo 3 >exp/frame_subsampling_factor

#     # [ -d $dir ] || mkdir -p $dir

#     # create decoding model
#     # disable dropout and batchnorm
#     # nnet3-am-copy --prepare-for-test=true $am ${dir}/final.mdl
#     # cp $am ${dir}/final.mdl

#     # copy tree
#     # cp ${src_dir}/tree $dir

#     ./utils/mkgraph.sh --self-loop-scale 1.0 $lang_test $dir $graph_dir || exit 1;
# fi

# decode_iter=1500

if [ $stage -le 3 ]; then
    echo "$0: decode into lattice"
    for decode_set in $test_sets; do

        # decode_dir=$dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall_merged_lv_hmhm
        decode_dir=$dir/decode_${decode_set}${decode_iter:+_$decode_iter}$lang_tag

        [ -d $decode_dir ] && rm -r $decode_dir

        # * multi process takes lot of memory
        steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
            --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
            --online-ivector-dir ${ivector_dir}/ivectors_${decode_set} \
            $graph_dir data/${decode_set}_hires \
            $decode_dir || exit 1;

        # cat $decode_dir/wer_* | utils/best_wer.sh
        ./local_bh/score_ali.sh --stats $stats_wer data/${decode_set}_hires $decode_dir
        cat $decode_dir/scoring/best_wer

    done
fi

if [ $stage -le 4 ] && $generate_ctm; then
    echo "$0: get timestamps"
    for decode_set in $test_sets; do
        # can also be lang, just contains words.txt and phones
        steps/conf/get_ctm_conf.sh data/$decode_set $graph_dir ${dir}/decode_${decode_set}${decode_iter:+_$decode_iter}_small_tg || exit 1
    done
fi
