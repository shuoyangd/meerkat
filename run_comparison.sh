#!/bin/bash

set -e

export PATH="/home/hxu/anaconda3/bin:$PATH"

mkdir -p extracted_soft_aligns/

# /home/shuoyangd/projects/realign/alg_exps/out_v3/align/*/out.[at|sa]

mkdir -p soft_align_aers/

false && for lang in deen enfr roen; do
  for i in `ls -l /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_soft/ | grep "\->" | grep $lang | awk '{print $(NF-2)}'`; do
    echo python soft_aer.py data2/$lang.talp /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_soft/$i/out.alg
    python soft_aer.py data2/$lang.talp /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_soft/$i/out.alg > soft_align_aers/$i.txt
  done
done

for lang in deen enfr roen; do
  cat data2/$lang.talp | sed "s=p=-=g" > data2/$lang.talp.nop
done

../fast_align/build/atools -c invert -i data2/deen.talp.nop > data2/ende.talp
../fast_align/build/atools -c invert -i data2/enfr.talp.nop > data2/fren.talp
../fast_align/build/atools -c invert -i data2/roen.talp.nop > data2/enro.talp

for lang in ende fren enro; do
  for i in `ls -l /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_soft/ | grep "\->" | grep $lang | awk '{print $(NF-2)}'`; do
    echo python soft_aer.py data2/$lang.talp /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_soft/$i/out.alg
    python soft_aer.py data2/$lang.talp /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_soft/$i/out.alg > soft_align_aers/$i.txt
  done
done

#             /home/shuoyangd/projects/realign/alg_exps/out_v3/debpe_ensemble_soft/*/out.alg                                                         

false && for i in `ls -l /home/shuoyangd/projects/realign/alg_exps/out_v3/align/ | grep -v LOCK | grep -v "\->" | grep -v "^total" | awk '{print $NF}'`; do
  echo i is $i
  python load_atts.py /home/shuoyangd/projects/realign/alg_exps/out_v3/align/$i/out.sa > extracted_soft_aligns/${i}_sa.txt
  python load_atts.py /home/shuoyangd/projects/realign/alg_exps/out_v3/align/$i/out.sa > extracted_soft_aligns/${i}_sa.txt
done

false && for i in out_v3_other_bp out_v3_tfm_bp out_v3_smooth; do
  for j in `ls -l /export/c11/shuoyangd/projects/realign/alg_exps/out_v3_other_bp/align/ | grep -v "\->" | awk '{print $NF}'`; do
    python load_atts.py /export/c11/shuoyangd/projects/realign/alg_exps/out_v3_other_bp/align/Arch.lstm/out.sa > extracted_soft_aligns/${i}_${j}_sa.txt
    python load_atts.py /export/c11/shuoyangd/projects/realign/alg_exps/out_v3_other_bp/align/Arch.lstm/out.at > extracted_soft_aligns/${i}_${j}_at.txt
  done
done
