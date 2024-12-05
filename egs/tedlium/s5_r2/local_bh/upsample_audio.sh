#!/usr/bin/env bash
# Copyright 2022  Bofeng Huang
# Upsample wavs in kaldi's wav.scp and rewrite a new one

# wavscp=/home/bhuang/corpus/speech/internal/hm_hm/train/wav.scp
# new_wavscp=/home/bhuang/corpus/speech/internal/hm_hm/train_16k/new_wav.scp
# wavscp=/home/bhuang/corpus/speech/internal/hm_hm/test/wav.scp
# new_wavscp=/home/bhuang/corpus/speech/internal/hm_hm/test_16k/new_wav.scp
wavscp=

out_wavdir=/home/bhuang/corpus/speech/internal/hm_hm/audios_16k

. ./utils/parse_options.sh

[ ! -d $out_wavdir ] && mkdir $out_wavdir

# 1. Create ProgressBar function
# 1.1 Input is currentState($1) and totalState($2)
function ProgressBar {
# Process data
    let _progress=(${1}*100/${2}*100)/100
    let _done=(${_progress}*4)/10
    let _left=40-$_done
# Build progressbar string lengths
    _fill=$(printf "%${_done}s")
    _empty=$(printf "%${_left}s")

# 1.2 Build progressbar strings and print the ProgressBar line
# 1.2.1 Output example:                           
# 1.2.1.1 Progress : [########################################] 100%
printf "\rProgress : [${_fill// /#}${_empty// /-}] ${_progress}%%"
}

line_counter=0
nlines=$(wc -l $wavscp)

# IFS= (or IFS='') prevents leading/trailing whitespace from being trimmed.
# -r prevents backslash escapes from being interpreted.
while IFS= read -r line; do
    # echo "Text read from file: $line"
    recording_id=$(echo $line | cut -d ' ' -f1)
    wavpath=$(echo $line | cut -d ' ' -f2)

    new_wavpath=$out_wavdir/$recording_id.wav

    # sox -G -v 0.95 $wavpath -r 16000 -c 1 -b 16 $new_wavpath
    # sox -G $wavpath -r 16000 -c 1 -b 16 $new_wavpath
    /home/bhuang/anaconda3/envs/asr/bin/python local_bh/upsample_audio.py \
        --in_wav_path $wavpath --out_wav_path $new_wavpath --in_sr 8000 --out_sr 16000

    # echo "$recording_id $new_wavpath" >> $new_wavscp

    line_counter=$((line_counter+1))
    ProgressBar $line_counter $nlines

done < $wavscp

echo "\nDone!"