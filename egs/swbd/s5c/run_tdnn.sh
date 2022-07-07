#!/usr/bin/env bash

# Copyright 2021  Zaion lab (Author: Bofeng Huang)

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model to a finetune data set.

# Usage: /home/daanzu/kaldi_dirs/local/run_finetune_tdnn_1a_daanzu.sh --src-dir export/tdnn_f.1ep --num-epochs 5 --stage 1 --train-stage -10

# Required Inputs:
#   data/finetune (text wav.scp utt2spk)
#   src_dir
#   tree_dir (tree final.mdl ali.*.gz phones.txt)
#   lang_dir (oov.int L.fst words.txt phones.txt phones/disambig.int)
#   conf_dir (mfcc.conf mfcc_hires.conf)
#   extractor_dir (final.ie final.dubm final.mat global_cmvn.stats splice_opts online_cmvn.conf online_cmvn_iextractor?)
# Writes To:
#   data/finetune, data/finetune_hires, data/finetune_sp, data/finetune_sp_hires,
#   exp/make_mfcc_chain/finetune, exp/make_mfcc_chain/finetune_sp_hires, exp/make_mfcc_chain/finetune_hires,
#   exp/nnet3_chain/ivectors_finetune_hires, exp/finetune_lats, exp/nnet3_chain/finetune

set -e

data_set=finetune
data_dir=data/${data_set}
conf_dir=conf
lang_dir=data/lang
extractor_dir=exp/nnet3_chain/extractor
# ali_dir=exp/${data_set}_ali
lat_dir=exp/${data_set}_lats
src_dir=exp/nnet3_chain/tdnn_f
tree_dir=exp/nnet3_chain/tree_sp
# dir=${src_dir}_${data_set}
dir=exp/nnet3_chain/${data_set}

num_gpus=1
num_epochs=5
# initial_effective_lrate=0.0005
# final_effective_lrate=0.00002
initial_effective_lrate=.00025
final_effective_lrate=.000025
minibatch_size=1024

xent_regularize=0.1
train_stage=-10
get_egs_stage=-10
common_egs_dir=  # you can set this to use previously dumped egs.
dropout_schedule='0,0@0.20,0.5@0.50,0'
frames_per_eg=150,110,100

stage=1
nj=24

echo "$0 $@"  # Print the command line for logging
. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

train_data_dir=${data_dir}_hires  # Containing perturbed data
train_ivector_dir=exp/nnet3_chain/ivectors_${data_set}_hires

if [ "$num_gpus" -eq 0 ]; then
  gpu_opt="no"
  num_gpus=1
else
  gpu_opt="wait"
fi

function log_stage () {
  echo
  echo "# Stage $1"
  echo "# $(date)"
  echo
}

# if [ $stage -le 1 ]; then
#   log_stage 1
#   utils/fix_data_dir.sh ${data_dir} || exit 1;
#   steps/make_mfcc.sh \
#     --cmd "$train_cmd" --nj $nj --mfcc-config $conf_dir/mfcc.conf \
#     ${data_dir} exp/make_mfcc_chain/${data_set} mfcc
# fi

if [ $stage -le 1 ]; then
  log_stage 1
  # (Approximately 0.66min single-core compute time per core per 1hr audio data)
  utils/fix_data_dir.sh ${data_dir} || exit 1;
  steps/make_mfcc.sh \
    --cmd "$train_cmd" --nj $nj --mfcc-config $conf_dir/mfcc.conf \
    ${data_dir} exp/make_mfcc_chain/${data_set} mfcc
  steps/compute_cmvn_stats.sh ${data_dir} exp/make_mfcc_chain/${data_set} mfcc || exit 1;
  utils/fix_data_dir.sh ${data_dir} || exit 1;

  utils/data/perturb_data_dir_speed_3way.sh ${data_dir} ${data_dir}_sp

  # steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/${train_set}_sp || exit 1;
  # steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  # utils/fix_data_dir.sh data/${train_set}_sp

  utils/copy_data_dir.sh ${data_dir}_sp ${data_dir}_sp_hires
  utils/data/perturb_data_dir_volume.sh ${data_dir}_sp_hires || exit 1;

  steps/make_mfcc.sh \
    --cmd "$train_cmd" --nj $nj --mfcc-config $conf_dir/mfcc_hires.conf \
    ${data_dir}_sp_hires exp/make_mfcc_chain/${data_set}_sp_hires mfcc
  steps/compute_cmvn_stats.sh ${data_dir}_sp_hires exp/make_mfcc_chain/${data_set}_sp_hires mfcc || exit 1;
  utils/fix_data_dir.sh ${data_dir}_sp_hires || exit 1;

  utils/copy_data_dir.sh ${data_dir} $train_data_dir
  rm -f $train_data_dir/{cmvn.scp,feats.scp}
  #utils/data/perturb_data_dir_volume.sh $train_data_dir || exit 1;
  steps/make_mfcc.sh \
    --cmd "$train_cmd" --nj $nj --mfcc-config $conf_dir/mfcc_hires.conf \
    $train_data_dir exp/make_mfcc_chain/${data_set}_hires mfcc
  steps/compute_cmvn_stats.sh $train_data_dir exp/make_mfcc_chain/${data_set}_hires mfcc
fi

# if false && [ $stage -le 2 ]; then
#   log_stage 2
#   # align new data(finetune set) with GMM, we probably replace GMM with NN later
#   steps/compute_cmvn_stats.sh ${data_dir} exp/make_mfcc_chain/${data_set} mfcc || exit 1;
#   utils/fix_data_dir.sh ${data_dir} || exit 1;
#   steps/align_si.sh --cmd "$train_cmd" --nj ${nj} ${data_dir} $lang_dir exp/tri3 ${ali_dir}
# fi

if [ $stage -le 2 ]; then
  log_stage 2
  # (Approximately 0.066min single-core compute time per core per 1hr audio data)
  # steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
  #   ${data_dir}_sp_hires $extractor_dir \
  #   exp/nnet3_chain/ivectors_${data_set}_sp_hires
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
    $train_data_dir $extractor_dir $train_ivector_dir
fi

if false && [ $stage -le 3 ]; then
  log_stage 3
  # Extract mfcc_hires for AM finetuning
  utils/copy_data_dir.sh ${data_dir} $train_data_dir
  rm -f $train_data_dir/{cmvn.scp,feats.scp}
  #utils/data/perturb_data_dir_volume.sh $train_data_dir || exit 1;
  steps/make_mfcc.sh \
    --cmd "$train_cmd" --nj $nj --mfcc-config $conf_dir/mfcc_hires.conf \
    $train_data_dir exp/make_mfcc_chain/${data_set}_hires mfcc_hires
  steps/compute_cmvn_stats.sh $train_data_dir exp/make_mfcc_chain/${data_set}_hires mfcc_hires
fi

if [ $stage -le 4 ]; then
  log_stage 4
  # Align new data(finetune set) with NN
  # (Approximately 0.085hr single-core compute time per core per 1hr audio data)
  # steps/nnet3/align.sh --cmd "$train_cmd" --nj ${nj} ${data_dir} $lang_dir ${src_dir} ${ali_dir}
  steps/nnet3/align_lats.sh --cmd "$train_cmd" --nj ${nj} \
    --acoustic-scale 1.0 \
    --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
    --online-ivector-dir $train_ivector_dir \
    $train_data_dir $lang_dir ${src_dir} ${lat_dir}
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
  log_stage 8
  $train_cmd $dir/log/generate_input_model.log \
    nnet3-am-copy --raw=true $src_dir/final.mdl $dir/input.raw
fi

if [ $stage -le 9 ]; then
  log_stage 9
  # we use chain model from source to generate lats for target and the
  # tolerance used in chain egs generation using this lats should be 1 or 2 which is
  # (source_egs_tolerance/frame_subsampling_factor)
  # source_egs_tolerance = 5
  chain_opts=(--chain.alignment-subsampling-factor=1 --chain.left-tolerance=1 --chain.right-tolerance=1)

  steps/nnet3/chain/train.py --stage $train_stage ${chain_opts[@]} \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 128,64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_gpus \
    --trainer.optimization.num-jobs-final $num_gpus \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change 2.0 \
    --use-gpu $gpu_opt \
    --cleanup.remove-egs false \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

# if [ $stage -le 9 ]; then
#   log_stage 9
#   steps/nnet3/train_dnn.py --stage=$train_stage \
#     --cmd="$decode_cmd" \
#     --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
#     --trainer.input-model $dir/input.raw \
#     --trainer.num-epochs $num_epochs \
#     --trainer.optimization.num-jobs-initial $num_jobs_initial \
#     --trainer.optimization.num-jobs-final $num_jobs_final \
#     --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
#     --trainer.optimization.final-effective-lrate $final_effective_lrate \
#     --trainer.optimization.minibatch-size $minibatch_size \
#     --feat-dir ${data_dir}_hires \
#     --lang $lang_dir \
#     --ali-dir ${ali_dir} \
#     --dir $dir || exit 1;
# fi