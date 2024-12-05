#!/usr/bin/env bash
# Copyright 2021  Bofeng Huang
# this is a basic script which generates lm

stage=0

PRUNE_THRESHOLD=1e-8
order=3

textdir=/home/bhuang/models/asr/lm/data
outdir=/home/bhuang/models/asr/lm/outputs

# cut -d ' ' -f 2- /home/bhuang/kaldi/egs/swbd/s5c/data/test_hm_hm_10h/text
dev_text=/home/bhuang/models/asr/lm/text_hmhm

. ./path.sh
. ./utils/parse_options.sh

if [ $stage -le 1 ]; then
    echo "Count ngrams"

    # NGRAM_COUNT=$SRILM_SRC/bin/i686-m64/ngram-count

    for subtextdir in $textdir/*/; do
        # (
        # strip trailing slash (if any)
        subtextdir="${subtextdir%/}"
        subdirname="${subtextdir##*/}"

        echo "Count text data in $subdirname"

        args=""

        # todo
        for source in $subtextdir/*_processed.csv; do
            filename="${source##*/}"
            filename="${filename%.*}"

            args=$args"-sort -order $order -text $source -write $outdir/local/$subdirname/counts/$filename-counts.gz "

            [ -d $outdir/local/$subdirname/counts ] || mkdir -p $outdir/local/$subdirname/counts
        done

        # echo $args | xargs --max-procs=4 -n 7 $NGRAM_COUNT
        echo $args | xargs --max-procs=4 -n 7 -t ngram-count
        # ) &
    done
    # wait
fi

if [ $stage -le 2 ]; then
    echo "Make individual language models"

    for suboutdir in $outdir/local/*/; do
        # (
        # strip trailing slash (if any)
        suboutdir="${suboutdir%/}"

        echo "Make lm of ${suboutdir##*/}"

        # N-grams up to order 5
        # Kneser-Ney smoothing
        # N-gram probability estimates at the specified order n are interpolated with lower-order estimates
        # include the unknown-word token as a regular word
        # pruning N-grams based on the specified threshold
        make-big-lm \
            -name $suboutdir/biglm $(for k in $suboutdir/counts/*.gz; do echo " -read $k "; done) \
            -lm $suboutdir/lm.gz \
            -max-per-file 100000000 \
            -order $order \
            -kndiscount \
            -interpolate \
            -unk \
            -prune $PRUNE_THRESHOLD
        # ) &
    done
    # wait
fi

if [ $stage -le 3 ]; then
    echo "Determine interpolation weights"

    for suboutdir in $outdir/local/*/; do
        # strip trailing slash (if any)
        suboutdir="${suboutdir%/}"

        ngram -debug 2 -order $order -unk -lm $suboutdir/lm.gz -ppl $dev_text >$suboutdir/lm.ppl
    done

    compute-best-mix $outdir/local/*/lm.ppl >$outdir/local/best-mix.ppl

    exit
fi

if [ $stage -le 4 ]; then
    echo "Combine the models"

    # DIRS=(afp_eng apw_eng cna_eng ltw_eng nyt_eng wpb_eng xin_eng)
    # LAMBDAS=(0.00631272 0.000647602 0.251555 0.0134726 0.348953 0.371566 0.00749238)
    DIRS=(fr_common_crawl_processed fr_europarl_processed fr_gutenberg_processed fr_news_commentary_processed fr_news_crawl_processed)
    # LAMBDAS=(0.639705 0.0483369 0.200637 0.0742303 0.0370902)
    LAMBDAS=(0.640601 0.0479738 0.20093 0.0743922 0.0361026)

    ngram -order $order -unk \
        -lm $outdir/local/${DIRS[0]}/lm.gz -lambda ${LAMBDAS[0]} \
        -mix-lm $outdir/local/${DIRS[1]}/lm.gz \
        -mix-lm2 $outdir/local/${DIRS[2]}/lm.gz -mix-lambda2 ${LAMBDAS[2]} \
        -mix-lm3 $outdir/local/${DIRS[3]}/lm.gz -mix-lambda3 ${LAMBDAS[3]} \
        -mix-lm4 $outdir/local/${DIRS[4]}/lm.gz -mix-lambda4 ${LAMBDAS[4]} \
        -write-lm $outdir/mixed_lm.gz
fi
exit

if [ $stage -le 5 ]; then
    echo "Convert to KenLM"

    $JOSHUA/src/joshua/decoder/ff/lm/kenlm/build_binary mixed_lm.gz mixed_lm.kenlm
fi
