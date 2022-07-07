#!/usr/bin/env bash

# Copyright 2021  Zaion lab (Author: Bofeng Huang)

# begin configuration section.
cmd=run.pl
min_lmwt=5
max_lmwt=17
word_ins_penalty=0.0,0.5,1.0
#end configuration section.

stats=true

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

data=$1
dir=$2

for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
  for lmwt in $(seq $min_lmwt $max_lmwt); do
    # adding /dev/null to the command list below forces grep to output the filename
    grep WER $dir/wer_${lmwt}_${wip} /dev/null
  done
done | utils/best_wer.sh  >& $dir/scoring/best_wer || exit 1

best_wer_file=$(awk '{print $NF}' $dir/scoring/best_wer)
best_wip=$(echo $best_wer_file | awk -F_ '{print $NF}')
best_lmwt=$(echo $best_wer_file | awk -F_ '{N=NF-1; print $N}')

if [ -z "$best_lmwt" ]; then
  echo "$0: we could not get the details of the best WER from the file $dir/wer_*.  Probably something went wrong."
  exit 1;
fi

if $stats; then
  mkdir -p $dir/scoring/wer_details
  echo $best_lmwt > $dir/scoring/wer_details/lmwt # record best language model weight
  echo $best_wip > $dir/scoring/wer_details/wip # record best word insertion penalty

  $cmd $dir/scoring/log/stats1.log \
    cat $dir/scoring/${best_lmwt}.${best_wip}.txt \| \
    align-text --special-symbol="'***'" ark:$dir/scoring/text.filt ark:- ark,t:- \|  \
    utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring/wer_details/per_utt \|\
      utils/scoring/wer_per_spk_details.pl $data/utt2spk \> $dir/scoring/wer_details/per_spk || exit 1;

  $cmd $dir/scoring/log/stats2.log \
    cat $dir/scoring/wer_details/per_utt \| \
    utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
    sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring/wer_details/ops || exit 1;

  $cmd $dir/scoring/log/wer_bootci.log \
    compute-wer-bootci --mode=present \
      ark:$dir/scoring/text.filt ark:$dir/scoring/${best_lmwt}.${best_wip}.txt \
      '>' $dir/scoring/wer_details/wer_bootci || exit 1;
fi

exit 0
