#!/usr/bin/env bash
# Copyright 2021  Bofeng Huang
# this is a basic script which generates dict dir


stage=0

train_set=train

# lexicon
lexicon=data/lexicon/montreal/FR.dict
# g2p model
g2p_model=data/lexicon/montreal/g2p_models/model.fst


. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 0 ]; then
  echo "$0: prepare oov"

  oov=data/$train_set/oov
  [ -d $oov ] && rm -r $oov
  mkdir -p $oov

  # get all vocab in corpus and sort
  cat data/$train_set/text | cut -d ' ' -f 2- | tee $oov/text_clean_lm | tr -s ' ' '\n' | sort | uniq >$oov/full_vocab
  echo "$(wc -l <$oov/full_vocab) words in $oov/full_vocab"

  # rm '<unk>' from vocab
  # this is added by cleanup
  sed '/<unk>/d' $oov/full_vocab >temp && mv temp $oov/full_vocab
  # awk '!/<unk>/' $oov/full_vocab > temp && mv temp $oov/full_vocab

  # get oov words
  # which are words existing in lexicon but not in vocab list
  awk 'NR==FNR{a[$1]++; next} !($1 in a)' $lexicon $oov/full_vocab | sort >$oov/oov_vocab
  # count oov words
  echo "number of oov's: $(wc -l <$oov/oov_vocab)"

  # generate oov words' phonetiques
  echo "generate pronunciation using phonetisaurus"
  phonetisaurus-apply --model=$g2p_model --word_list $oov/oov_vocab >$oov/oov_lexicon

  # check if have added phones into lexicon
  # get all phones in oov_lexicon
  awk '{$1=""; print $0}' $oov/oov_lexicon | sed -e 's/^ //' | tr -s ' ' '\n' | sort -u >$oov/oov_phones.txt
  # get all phones in original lexicon
  awk '{$1=""; print $0}' $lexicon | sed -e 's/^ //' | tr -s ' ' '\n' | sort -u >$oov/fr_dict_phones.txt
  # check if exist phones in oov_lexicon but not in original lexicon
  if [[ $(comm -13 $oov/fr_dict_phones.txt $oov/oov_phones.txt) ]]; then
    echo "WARNING : phonetisaurus-apply added extra phones for oovs in $x!!"
    exit 1
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: prepare dict"

  dict=data/local/dict_nosp
  
  mkdir -p $dict
  lexicon_raw_nosil=$dict/lexicon_raw_nosil.txt
  silence_phones=$dict/silence_phones.txt
  optional_silence=$dict/optional_silence.txt
  nonsil_phones=$dict/nonsilence_phones.txt
  extra_questions=$dict/extra_questions.txt

  # if [[ ! -s "$lexicon_raw_nosil" ]]; then
  # combine vocab lex and train oov lex
  # do not add test oov, or else it will be cheating !!
  cat $lexicon data/$train_set/oov/oov_lexicon | sort -k1 >$lexicon_raw_nosil
  #fi
  echo "Original lexicon has $(wc -l <$lexicon) words"
  echo "OOV lexicon has $(wc -l <data/$train_set/oov/oov_lexicon) words"
  # echo "Total vocab : $(wc -l <$lexicon_raw_nosil)"

  echo "Preparing phone lists and clustering questions"
  # create silence_phones.txt
  echo -e 'SIL'\\n'SPN' >$silence_phones

  # create optional_silence.txt
  echo 'SIL' >$optional_silence

  # create nonsilence_phones.txt
  # on each line is a list of phones that correspond
  # really to the same base phone.
  # todo: understand this shit
  awk '{for (i=2; i<=NF; ++i) {print $i; gsub(/[0-9]/, "", $i); print $i}}' $lexicon_raw_nosil |
    sort -u |
    perl -e 'while(<>){
    chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_";
    $phones_of{$1} .= "$_ "; }
    foreach $list (values %phones_of) {print $list . "\n"; } ' | sort \
    >$nonsil_phones || exit 1

  # create extra_questions.txt
  # A few extra questions that will be added to those obtained by automatically clustering the "real" phones.
  # These ask about stress; there's also one for silence.
  cat $silence_phones | awk '{printf("%s ", $1);} END{printf "\n";}' >$extra_questions || exit 1
  cat $nonsil_phones | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
    >>$extra_questions || exit 1

  # print file sizes
  echo "$(wc -l <$silence_phones) silence phones saved to: $silence_phones"
  echo "$(wc -l <$optional_silence) optional silence saved to: $optional_silence"
  echo "$(wc -l <$nonsil_phones) non-silence phones saved to: $nonsil_phones"
  echo "$(wc -l <$extra_questions) extra triphone clustering-related questions saved to: $extra_questions"

  # complete lexicon with SIL and SPN
  (
    echo '!SIL SIL'
    echo '<unk> SPN'
  ) |
    cat - $lexicon_raw_nosil | sort | uniq >$dict/lexicon.txt
  echo "$(wc -l <$dict/lexicon.txt) words saved to: $dict/lexicon.txt"
fi
