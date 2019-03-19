#!/bin/bash

mkdir -p data2/train
mkdir -p data2/test
mkdir -p models2/

# cp /home/shuoyangd/projects/realign/alignment-scripts/preprocess/train/{deen,enfr,roen}.lc.{src,tgt} data2/train
# cp /home/shuoyangd/projects/realign/alignment-scripts/preprocess/test/{deen,enfr,roen}.lc.{src,tgt} data2/test
# cp /home/shuoyangd/projects/realign/alignment-scripts/preprocess/test/{deen,enfr,roen}.talp data2/

# step-1, data preparation

for lang in deen enfr roen; do
#  paste data2/train/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data2/train.all.$lang
#  paste data2/test/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data2/test.all.$lang
#  cat data2/test.all.$lang >> data2/train.all.$lang
#
#  echo align forward...
#  ../fast_align/build/fast_align -i data2/train.all.$lang    -d -o -v -p models2/$lang.fw.mdl > models2/$lang.fw.align 2>models2/$lang.fw.err
#  echo align backward..
#  ../fast_align/build/fast_align -i data2/train.all.$lang -r -d -o -v -p models2/$lang.bw.mdl > models2/$lang.bw.align 2>models2/$lang.bw.err
#
  n=`cat data2/test/$lang.lc.src | wc -l | awk '{print $1}'`

  tail -n $n models2/$lang.fw.align > tmp.fw
  tail -n $n models2/$lang.bw.align > tmp.bw
  ../fast_align/build/atools -c invert -i tmp.bw > tmp.bw2


  echo lang is $lang
  python aer.py --oneRef data2/$lang.talp tmp.fw
  python aer.py --oneRef data2/$lang.talp tmp.bw

  python3 combine_bidirectional_alignments.py --method grow-diagonal tmp.fw tmp.bw2 > hyp.grow-diagonal
#  ../fast_align/build/atools -i tmp.fw -j tmp.bw -c grow-diag > tmp.hyp.txt
  python aer.py data2/$lang.talp hyp.grow-diagonal --oneRef

  python3 combine_bidirectional_alignments.py --method grow-diagonal-final tmp.fw tmp.bw2 > hyp.grow-diagonal-final
#  ../fast_align/build/atools -i tmp.fw -j tmp.bw -c grow-diag-final > tmp.hyp.txt
  python aer.py data2/$lang.talp hyp.grow-diagonal-final --oneRef


##  echo align test......
#  ../fast_align/build/force_align.py models/$lang.fw.mdl models/$lang.fw.err models/$lang.bw.mdl models/$lang.bw.err grow-diag-final-and <data/test.all.$lang >models/test.$lang.align.grow-diag-final-and
#  ../fast_align/build/force_align.py models/$lang.fw.mdl models/$lang.fw.err models/$lang.fw.mdl models/$lang.fw.err union <data/test.all.$lang >models/test.$lang.align.forward
#  ../fast_align/build/force_align.py models/$lang.bw.mdl models/$lang.bw.err models/$lang.bw.mdl models/$lang.bw.err union <data/test.all.$lang >models/test.$lang.align.backward
#
#  echo lang is $lang
#  python aer.py --oneRef data/$lang.talp models/test.$lang.align.grow-diag-final-and
#  python aer.py --oneRef data/$lang.talp models/test.$lang.align.forward
#  python aer.py --oneRef data/$lang.talp models/test.$lang.align.backward
#
##  paste data/train/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data/train.all.$lang
##  paste data/test/$lang.lc.{src,tgt} | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > data/test.all.$lang
##
##  echo align forward...
##  ../fast_align/build/fast_align -i data/train.all.$lang    -d -o -v -p models/$lang.fw.mdl > models/$lang.fw.align 2>models/$lang.fw.err
##  echo align backward..
##  ../fast_align/build/fast_align -i data/train.all.$lang -r -d -o -v -p models/$lang.bw.mdl > models/$lang.bw.align 2>models/$lang.bw.err
##
##  echo align test......
##  ../fast_align/build/force_align.py models/$lang.fw.mdl models/$lang.fw.err models/$lang.bw.mdl models/$lang.bw.err <data/test.all.$lang >models/test.$lang.align
##
##  python aer.py --oneRef data/$lang.talp models/test.$lang.align

done
