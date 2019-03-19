#!/bin/bash

mkdir -p data/train
mkdir -p data/test
mkdir -p models/

# cp /home/shuoyangd/projects/realign/alignment-scripts/preprocess/train/{deen,enfr,roen}.lc.{src,tgt} data/train
# cp /home/shuoyangd/projects/realign/alignment-scripts/preprocess/test/{deen,enfr,roen}.lc.{src,tgt} data/test
# cp /home/shuoyangd/projects/realign/alignment-scripts/preprocess/test/{deen,enfr,roen}.talp data/

# step-1, data preparation

for lang in deen enfr roen; do
#  paste data/train/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data/train.all.$lang
#  paste data/test/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data/test.all.$lang
##  cat data/test.all.$lang >> data/train.all.$lang
#
#  echo align forward...
#  ../fast_align/build/fast_align -i data/train.all.$lang    -d -o -v -p models/$lang.fw.mdl > models/$lang.fw.align 2>models/$lang.fw.err
#  echo align backward..
#  ../fast_align/build/fast_align -i data/train.all.$lang -r -d -o -v -p models/$lang.bw.mdl > models/$lang.bw.align 2>models/$lang.bw.err

##  n=`cat data/test/$lang.lc.src | wc -l | awk '{print $1}'`
##  echo test has $n lines
##
##  tail -n $n models/$lang.fw.align > tmp.fw
##  tail -n $n models/$lang.bw.align > tmp.bw
##
##  ../fast_align/build/atools -i tmp.fw -j tmp.bw -c grow-diag-final-and > tmp.hyp.txt
##  python aer.py --oneRef data/$lang.talp tmp.hyp.txt

#  echo align test......

#  ../fast_align/build/force_align.py          models/$lang.fw.mdl models/$lang.fw.err models/$lang.bw.mdl models/$lang.bw.err grow-diag-final <data/test.all.$lang >models/test.$lang.align.grow-diag-final-atool
  ../fast_align/build/force_align_forward.py  models/$lang.fw.mdl models/$lang.fw.err models/$lang.fw.mdl models/$lang.fw.err union <data/test.all.$lang >models/test.$lang.align.forward
  ../fast_align/build/force_align_backward.py models/$lang.bw.mdl models/$lang.bw.err models/$lang.bw.mdl models/$lang.bw.err union <data/test.all.$lang >models/test.$lang.align.backward

  ../fast_align/build/atools -c grow-diag-final -i models/test.$lang.align.forward -j models/test.$lang.align.backward > models/test.$lang.align.grow-diag-final-atool
  ../fast_align/build/atools -c invert -i models/test.$lang.align.backward > models/test.$lang.align.backward2
  python3 combine_bidirectional_alignments.py --method grow-diagonal-final models/test.$lang.align.forward models/test.$lang.align.backward2 >models/test.$lang.align.grow-diag-final

  echo lang is $lang
  python aer.py --oneRef data/$lang.talp models/test.$lang.align.forward
  python aer.py --oneRef data/$lang.talp models/test.$lang.align.backward
  python aer.py --oneRef data/$lang.talp models/test.$lang.align.grow-diag-final
  python aer.py --oneRef data/$lang.talp models/test.$lang.align.grow-diag-final-atool

#  paste data/train/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data/train.all.$lang
#  paste data/test/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data/test.all.$lang
#
#  echo align forward...
#  ../fast_align/build/fast_align -i data/train.all.$lang    -d -o -v -p models/$lang.fw.mdl > models/$lang.fw.align 2>models/$lang.fw.err
#  echo align backward..
#  ../fast_align/build/fast_align -i data/train.all.$lang -r -d -o -v -p models/$lang.bw.mdl > models/$lang.bw.align 2>models/$lang.bw.err
#
#  echo align test......
#  ../fast_align/build/force_align.py models/$lang.fw.mdl models/$lang.fw.err models/$lang.bw.mdl models/$lang.bw.err <data/test.all.$lang >models/test.$lang.align
#
#  python aer.py --oneRef data/$lang.talp models/test.$lang.align

done
