#!/bin/bash

export PATH="/home/hxu/anaconda3/bin:$PATH"

#/home/shuoyangd/projects/realign/alg_exps/out_v3/align/*/out.{at,sa}
#
#/home/shuoyangd/projects/realign/alg_exps/out_v3_free/align/*/out.{at,sa}

mkdir -p entropies/
dir=entropies

for i in `ls -l /home/shuoyangd/projects/realign/alg_exps/out_v3/align | grep -v "\->" | awk '{print $NF}'`; do
  [ ! -f /home/shuoyangd/projects/realign/alg_exps/out_v3/align/$i/out.sa ] && continue
  python load_and_entropy.py /home/shuoyangd/projects/realign/alg_exps/out_v3/align/$i/out.sa > $dir/$i.sa.txt
  python load_and_entropy.py /home/shuoyangd/projects/realign/alg_exps/out_v3/align/$i/out.at > $dir/$i.at.txt
done

for i in `ls -l /home/shuoyangd/projects/realign/alg_exps/out_v3_free/align | grep -v "\->" | awk '{print $NF}'`; do
  [ ! -f /home/shuoyangd/projects/realign/alg_exps/out_v3_free/align/$i/out.sa ] && continue
  python load_and_entropy.py /home/shuoyangd/projects/realign/alg_exps/out_v3_free/align/$i/out.sa > $dir/$i.free.sa.txt
  python load_and_entropy.py /home/shuoyangd/projects/realign/alg_exps/out_v3_free/align/$i/out.sa > $dir/$i.free.sa.txt
done
