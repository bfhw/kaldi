#!/usr/bin/env bash

# speechbrain csv to kaldi data

echo "$0 $@" # Print the command line for logging

KALDI_ROOT=/home/bhuang/kaldi

csvfile=$1
outdir=$2

[ -d $outdir ] || mkdir -p $outdir

tail -n +2 $csvfile | awk -F, 'function basename(name) {sub("\-[0-9]{3}", "", name); return name} {print $1, basename($1)}' | sort -u -k1,1 >${outdir}/utt2spk

tail -n +2 $csvfile | awk -F, '{print $1, $5}' | sort -u -k1,1 >${outdir}/text

tail -n +2 $csvfile | awk -F, 'function basename(name) {sub("\-[0-9]{3}", "", name); return name} {print basename($1), $6}' | sort -u -k1,1 >${outdir}/wav.scp

tail -n +2 $csvfile | awk -F, 'function basename(name) {sub("\-[0-9]{3}", "", name); return name} {print $1, basename($1), $2, $3}' | sort -u -k1,1 >${outdir}/segments


# fix data
${KALDI_ROOT}/egs/wsj/s5/utils/fix_data_dir.sh $outdir || exit 1;
