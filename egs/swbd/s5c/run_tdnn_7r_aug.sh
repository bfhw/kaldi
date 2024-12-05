#!/usr/bin/env bash
# Copyright 2021  Bofeng Huang
# Recipe for training tdnn-f models on hires features using data augmentation

# this is based on
# s5c/local/chain/tuning/run_tdnn_7r.sh (traditional delta and delta-delta features + sp perturbation)
# s5c/local/chain/multi_condition/run_tdnn_aug_1a.sh (run_aug)

set -euo pipefail
# set -e

export CUDA_VISIBLE_DEVICES=3

# configs for 'chain'
stage=1
nj=50

train_stage=-10
get_egs_stage=-10

# speed perturbation
speed_perturb=true

# multi condition augmentation
multi_condition=true
# aug_list="reverb music noise babble clean"
# bh: empirically best option on hmhm
aug_list="reverb noise clean"
num_reverb_copies=1
use_ivectors=true
# multi condition used corpus
musan_corpus=/home/bhuang/corpus/speech/public/musan
rirs_corpus=/home/bhuang/corpus/speech/public/RIRS_NOISES

# ? when utt are too short
# frames_per_eg=75,50,25
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

frame_subsampling_factor=3

# training options
num_epochs=6
# num_jobs_initial=2
# num_jobs_final=4
num_jobs_initial=1
num_jobs_final=1
# initial_effective_lrate=0.0005
# final_effective_lrate=0.00002
initial_effective_lrate=.00025
final_effective_lrate=.000025

# decoding
decode_iter=
decode_nj=50

# if true, it will run the last decoding stage.
test_online_decoding=false

# * empirically best silence setup for GMM
boost_sil_opts=(--boost-silence 1.25)

# data
clean_set=train_hmhm190h
test_sets="test_hmhm10h"

echo "$0 $@" # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

affix=7r

suffix=
$multi_condition && suffix=_aug${suffix}
$speed_perturb && suffix=_sp${suffix}

# lang
lang=data/lang_chain_2y

# ali, lat
# ali_model=tri3
ali_model=tdnn_hmhm190h
clean_ali=${ali_model}_ali_nodup
ali_dir=$clean_ali$suffix
clean_lat=${ali_model}_lats_nodup
lat_dir=$clean_lat$suffix

# tree
treedir=exp/chain/tri5_7d_tree${suffix}

# outdir
dir=exp/chain/tdnn${affix}${suffix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# if [ "$num_gpus" -eq 0 ]; then
if [ "$num_jobs_initial" -eq 0 ]; then
  gpu_opt="no"
  # num_gpus=1
else
  gpu_opt="wait"
fi

function log_stage() {
  echo
  echo "# Stage $1: $2"
  echo "# $(date)"
  echo
}

if [ $stage -le 0 ]; then
  log_stage 0 "Extract mfcc features"

  mfccdir=mfcc
  if [ ! -f data/$clean_set/feats.scp ]; then
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$clean_set exp/make_mfcc/$clean_set $mfccdir || exit 1
    steps/compute_cmvn_stats.sh data/$clean_set exp/make_mfcc/$clean_set $mfccdir || exit 1
    utils/fix_data_dir.sh data/$clean_set
  else
    echo "MFCC features of $clean_set already exist. Maybe you want to regenerate them?"
    exit
  fi
fi

if [ $stage -le 0 ]; then
  log_stage 0 "Remove the duplicate utterances"

  # utils/subset_data_dir.sh --shortest data/$clean_set 100000 data/${clean_set}_100kshort
  # utils/subset_data_dir.sh data/${clean_set}_100kshort 30000 data/${clean_set}_30kshort

  utils/subset_data_dir.sh --first data/$clean_set 100000 data/${clean_set}_100k
  utils/data/remove_dup_utts.sh 200 data/${clean_set}_100k data/${clean_set}_100k_nodup

  utils/data/remove_dup_utts.sh 300 data/${clean_set} data/${clean_set}_nodup
fi

ivector_trainset=${clean_set}_100k_nodup
clean_set=${clean_set}_nodup
train_set=${clean_set}$suffix

# normally done in previous gmm training
if [ $stage -le 1 ]; then
  log_stage 1 "Obtain the alignment of the ivector training data"

  # todo: --use-ivectors "$use_ivectors" \
  # use tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${boost_sil_opts[@]} \
    data/$ivector_trainset data/lang \
    exp/tri1b exp/tri1b_ali_$ivector_trainset
fi

if $multi_condition; then
  # Here we recommend speed perturbation as the gains are significant.
  # The gain from speed perturbation is additive with the gain from data reverberation
  if $speed_perturb; then
    if [ $stage -le 2 ]; then
      # Although the nnet will be trained by high resolution data, we still have
      # to perturb the normal data to get the alignments _sp stands for
      # speed-perturbed
      log_stage 2 "Preparing directory for speed-perturbed data"

      utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
        data/${clean_set} data/${clean_set}_sp
    fi

    if [ $stage -le 3 ]; then
      log_stage 3 "Creating MFCC features for low-resolution speed-perturbed data"

      mfccdir=mfcc_perturbed
      steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${clean_set}_sp exp/make_mfcc/${clean_set}_sp $mfccdir
      steps/compute_cmvn_stats.sh data/${clean_set}_sp exp/make_mfcc/${clean_set}_sp $mfccdir
      utils/fix_data_dir.sh data/${clean_set}_sp
    fi

    if [ $stage -le 4 ]; then
      # if we are using the speed-perturbed data we need to generate alignments for it.
      log_stage 4 "Obtain the alignment of the speed perturbated data"

      # * gmm ali
      tmp_nj=$(cat exp/${ali_model}/num_jobs) || exit 1
      steps/align_fmllr.sh --nj $tmp_nj --cmd "$train_cmd" \
        ${boost_sil_opts[@]} \
        data/${clean_set}_sp data/lang exp/${ali_model} exp/${clean_ali}_sp

      # * tdnn ali
      # ali_model=/projects/bhuang/models/asr/kaldi/tdnn7r_sp_aug_hmhm190h/tdnn7r_sp_aug_hmhm190h
      # ali_ivector_model=/projects/bhuang/models/asr/kaldi/tdnn7r_sp_aug_hmhm190h/nnet3_tdnn7r_sp_aug_hmhm190h
      # tmp_nj=$(cat ${ali_model}/num_jobs) || exit 1

      # utils/copy_data_dir.sh data/${clean_set}_sp data/${clean_set}_sp_hires_for_ali

      # mfccdir=mfcc_hires
      # steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
      #   --mfcc-config conf/mfcc_hires.conf \
      #   data/${clean_set}_sp_hires_for_ali exp/make_mfcc/${clean_set}_sp_hires_for_ali $mfccdir || exit 1;
      # steps/compute_cmvn_stats.sh data/${clean_set}_sp_hires_for_ali exp/make_mfcc/${clean_set}_sp_hires_for_ali $mfccdir
      # utils/fix_data_dir.sh data/${clean_set}_sp_hires_for_ali || exit 1;

      # # extract ivector
      # # ? --utts-per-spk-max 2
      # steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      #   data/${clean_set}_sp_hires_for_ali ${ali_ivector_model}/extractor ${ali_ivector_model}/ivectors_pre_${clean_set}_sp_hires_for_ali|| exit 1;

      # # use tdnn ali to build tree
      # steps/nnet3/align.sh --nj $tmp_nj --cmd "$train_cmd" \
      #   --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
      #   --online_ivector_dir ${ali_ivector_model}/ivectors_pre_${clean_set}_sp_hires_for_ali \
      #   data/${clean_set}_sp_hires_for_ali data/lang $ali_model exp/${clean_ali}_sp || exit 1

    fi

    clean_set=${clean_set}_sp
    clean_ali=${clean_ali}_sp
    clean_lat=${clean_lat}_sp
  fi

  if [ $stage -le 5 ]; then
    log_stage 5 "Perturbe the training data with multi conditions, train and extract the ivectors"

    # bh: duplicate ali into ${clean_ali}_aug
    # bh: train ivector on mutli conditions but not speed
    # local/nnet3/multi_condition/run_aug_common.sh \
    local_bh/run_aug_common.sh \
      --aug-list "$aug_list" \
      --num-reverb-copies $num_reverb_copies \
      --use-ivectors "$use_ivectors" \
      --train-set $clean_set \
      --test-sets $test_sets \
      --clean-ali $clean_ali \
      --ivector-trainset $ivector_trainset \
      --lda-mllt-ali tri1b_ali_$ivector_trainset \
      --musan-corpus $musan_corpus \
      --rirs-corpus $rirs_corpus \
      --nj $nj || exit 1
  fi

  if [ $stage -le 6 ]; then
    log_stage 6 "Obtain the lattice of the augmentated data"

    # Get the alignments as lattices (gives the LF-MMI training more freedom).
    # use the same num-jobs as the alignments
    prefixes=""
    include_original=false
    for n in $aug_list; do
      if [ "$n" == "reverb" ]; then
        for i in $(seq 1 $num_reverb_copies); do
          prefixes="$prefixes "reverb$i
        done
      elif [ "$n" != "clean" ]; then
        prefixes="$prefixes "$n
      else
        # The original train directory will not have any prefix
        # include_original flag will take care of copying the original lattices
        include_original=true
      fi
    done

    # * gmm lat
    tmp_nj=$(cat exp/$ali_dir/num_jobs) || exit 1
    steps/align_fmllr_lats.sh --nj $tmp_nj --cmd "$train_cmd" \
      data/${clean_set} data/lang exp/${ali_model} exp/${clean_lat}
    rm exp/${clean_lat}/fsts.*.gz # save space

    # * tdnn lat
    # ali_model=/projects/bhuang/models/asr/kaldi/tdnn7r_sp_aug_hmhm190h/tdnn7r_sp_aug_hmhm190h
    # ali_ivector_model=/projects/bhuang/models/asr/kaldi/tdnn7r_sp_aug_hmhm190h/nnet3_tdnn7r_sp_aug_hmhm190h
    # tmp_nj=$(cat ${ali_model}/num_jobs) || exit 1

    # steps/nnet3/align_lats.sh --cmd "$train_cmd" --nj $tmp_nj \
    #   --acoustic-scale 1.0 \
    #   --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
    #   --generate-ali-from-lats true \
    #   --online_ivector_dir ${ali_ivector_model}/ivectors_pre_${clean_set}_hires_for_ali \
    #   data/${clean_set}_hires_for_ali data/lang $ali_model exp/${clean_lat} || exit 1

    # bh: These lattices are used in get_egs.sh to generate the training examples
    steps/copy_lat_dir.sh --nj $tmp_nj --cmd "$train_cmd" \
      --include-original "$include_original" --prefixes "$prefixes" \
      data/${train_set} exp/${clean_lat} exp/${lat_dir} || exit 1
  fi
else
  if [ $stage -le 7 ]; then
    log_stage 7 "Perturbe speed and volume of the training data, train and extract the ivectors"

    # if we are using the speed-perturbed data we need to generate
    # alignments for it.

    # todo: --use-ivectors "$use_ivectors" \
    # use tri+sat as ali
    # gen ali to: exp/${ali_model}_ali_nodup_sp
    # gen ivectors to: exp/nnet3/ivectors_$data_set
    # local/nnet3/run_ivector_common.sh \
    local_bh/run_ivector_common.sh \
      ${boost_sil_opts[@]} \
      --speed-perturb $speed_perturb \
      --generate-alignments $speed_perturb \
      --train-set ${clean_set} \
      --ivector-trainset $ivector_trainset \
      --test-sets ${test_sets} \
      --ali-model ${ali_model} \
      --ivector-trainset-ali tri1b_ali_$ivector_trainset || exit 1
  fi

  if [ $stage -le 8 ]; then
    log_stage 8 "Obtain the lattices of the speed perturbated data"

    # bh: build tree use ali, training use lats
    # bh: These lattices are used in get_egs.sh to generate the training examples
    # Get the alignments as lattices (gives the LF-MMI training more freedom).
    # use the same num-jobs as the alignments
    tmp_nj=$(cat exp/${ali_dir}/num_jobs) || exit 1
    steps/align_fmllr_lats.sh --nj $tmp_nj --cmd "$train_cmd" \
      ${boost_sil_opts[@]} \
      data/${train_set} data/lang exp/tri3 exp/${lat_dir} || exit 1
    rm exp/${lat_dir}/fsts.*.gz # save space
  fi
fi

if [ $stage -le 10 ]; then
  log_stage 10 "Generate topo"

  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]

  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  log_stage 11 "Generate tree"

  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  # ? num_leaves
  # * for gmm ali
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor $frame_subsampling_factor \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 7000 data/$train_set $lang exp/$ali_dir $treedir

  # * for tdnn ali
  # bh: alignment-subsampling-factor no need for gmm ali but for tdnn
  # steps/nnet3/chain/build_tree.sh \
  # --frame-subsampling-factor $frame_subsampling_factor \
  # --alignment-subsampling-factor 1 \
  # --context-opts "--context-width=2 --central-position=1" \
  # --cmd "$train_cmd" 7000 data/$train_set $lang exp/$ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  # echo "$0: creating neural net configs using the xconfig parser"
  log_stage 12 "creating neural net configs using the xconfig parser"

  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF >$dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  delta-layer name=delta
  no-op-component name=input2 input=Append(delta, Scale(1.0, ReplaceIndex(ivector, t, 0)))

  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536 input=input2
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

# * cnn-tdnn model
# if [ $stage -le 12 ]; then
#   echo "$0: creating neural net configs using the xconfig parser";

#   num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
#   learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

#   cnn_opts="l2-regularize=0.01"
#   ivector_affine_opts="l2-regularize=0.01"
#   tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
#   tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
#   linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
#   prefinal_opts="l2-regularize=0.01"
#   output_opts="l2-regularize=0.002"

#   mkdir -p $dir/configs
#   cat <<EOF > $dir/configs/network.xconfig
#   input dim=100 name=ivector
#   input dim=40 name=input
#   # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
#   # are more compressible so we prefer to dump the MFCCs to disk rather
#   # than filterbanks.
#   idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
#   linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
#   batchnorm-component name=ivector-batchnorm target-rms=0.025
#   batchnorm-component name=idct-batchnorm input=idct
#   combine-feature-maps-layer name=combine_inputs input=Append(idct-batchnorm, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40
#   conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
#   conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
#   conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
#   conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
#   conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
#   conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10  time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
#   # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
#   # information bottleneck doesn't become a problem.  (we use time-stride=0 so no splicing, to
#   # limit the num-parameters).
#   tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
#   tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
#   linear-component name=prefinal-l dim=256 $linear_opts
#   ## adding the layers for chain branch
#   prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
#   output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
#   # adding the layers for xent branch
#   prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
#   output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
# EOF
#   steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
# fi

if [ $stage -le 13 ]; then
  log_stage 13 "Train nnet from scratch"

  # * for tdnn ali
  # chain_opts=(--chain.alignment-subsampling-factor=1 --chain.frame-subsampling-factor=$frame_subsampling_factor)
  # todo
  # chain_opts=(--chain.alignment-subsampling-factor=1 --chain.frame-subsampling-factor=$frame_subsampling_factor --chain.left-tolerance 3 --chain.right-tolerance 3)

  # --trainer.num-chunk-per-minibatch 128,64 \
  # --trainer.frames-per-iter 1500000 \
  # num-jobs-initial -> num-jobs-final, lr, max_change
  steps/nnet3/chain/train.py --stage $train_stage \
    ${chain_opts[@]} \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
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
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --use-gpu $gpu_opt \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/$lat_dir \
    --dir $dir || exit 1
fi

if [ $stage -le 14 ]; then
  log_stage 14 "Compile the decoding graph"
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.

  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_tgsmall $dir $dir/graph_tgsmall

  # ! try another lang
  # lang_test=data/lang_nosp_test_tgsmall_merged_lv_hmhm
  # graph_dir=$dir/graph_tgsmall_merged_lv_hmhm
  # utils/mkgraph.sh --self-loop-scale 1.0 $lang_test $dir $graph_dir
fi

iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  log_stage 15 "Decode"

  rm $dir/.error 2>/dev/null || true

  graph_dir=$dir/graph_tgsmall
  # ! try another lang
  # graph_dir=$dir/graph_tgsmall_merged_lv_hmhm

  for decode_set in $test_sets; do
    (
      decode_dir=$dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall
      # ! try another lang
      # decode_dir=$dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall_merged_lv_hmhm

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires \
        $decode_dir || exit 1

      # if $has_fisher; then
      #     steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      #       data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
      #       $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      # fi

      ./local_bh/score_ali.sh --stats false data/${decode_set}_hires $decode_dir
      cat $decode_dir/scoring/best_wer

    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000 $maybe_rt03; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
        --acwt 1.0 --post-decode-acwt 10.0 \
        $graph_dir data/${decode_set}_hires \
        ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1
      if $has_fisher; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
          ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
