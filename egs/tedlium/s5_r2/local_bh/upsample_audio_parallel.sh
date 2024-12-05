#!/usr/bin/env bash
# Copyright 2022  Bofeng Huang
# A basic script to run something in parallel with kaldi's run.pl script

# ./path.sh
# utils/parse_options.sh || exit 1;

raw_file=$1
outdir=$2

tmpdir=$outdir/tmp

njob=40

stage=1

if [ $stage -le 0 ]; then
    echo "$0: split files"

    # ls $audiodir >$tmpdir/audios_lst.txt

    total_lines=$(wc -l <$raw_file)
    lines_per_file=$((($total_lines + $njob - 1) / $njob))

    # echo "$total_lines"
    # echo "$lines_per_file"

    [ -d $tmpdir/splits ] || mkdir -p $tmpdir/splits

    split --numeric-suffixes=1 -l $lines_per_file $raw_file $tmpdir/splits/split_
    rename 's/_0{1,}([0-9]+)$/_$1/' $tmpdir/splits/split_* || exit 1
fi

if [ $stage -le 1 ]; then
    echo "$0: run in parallel"

    [ -d $tmpdir/log ] || mkdir -p $tmpdir/log
    # [ -d $outdir/audios ] || mkdir -p $outdir/audios
    # [ -d $outdir/results ] || mkdir -p $outdir/results

    # /home/it-zaion/kaldi/egs/wsj/s5/utils/run.pl JOB=1:$njob $tmpdir/log/run_log.JOB.log /home/it-zaion/extr-asr-test-data/examples_stt/ms/tmp_ms_stt.sh $tmpdir/splits/split_JOB $outdir/audios $outdir/results || exit 1
    /home/bhuang/kaldi/egs/wsj/s5/utils/run.pl JOB=1:$njob $tmpdir/log/run_log.JOB.log /home/bhuang/kaldi/egs/tedlium/s5_r2/local_bh/upsample_audio.sh --wavscp $tmpdir/splits/split_JOB || exit 1
fi
