#!/bin/bash

# /home/shuoyangd/projects/realign/*_exps/out/decode/*/out
mkdir -p all_align

false && for lang in deen enfr roen; do
  echo Processing $lang
  mkdir -p all_align/$lang
  base=/home/shuoyangd/projects/realign/${lang}_exps/out/decode/
  for i in 4F9EC56BCB7FD68922D70E08FEBFA1B5 Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none; do
    echo ... Processing $i
    cat $base/$i/out | sed "s= ==g" | sed "s=▁= =g" > all_align/$lang/$i.txt

    paste data/test/$lang.lc.src all_align/$lang/$i.txt | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > all_align/$lang/$i.to_align

    ../fast_align/build/force_align_forward.py  models/$lang.fw.mdl models/$lang.fw.err models/$lang.fw.mdl models/$lang.fw.err union <all_align/$lang/$i.to_align >all_align/$lang/$i.align.forward
    ../fast_align/build/force_align_backward.py models/$lang.bw.mdl models/$lang.bw.err models/$lang.bw.mdl models/$lang.bw.err union <all_align/$lang/$i.to_align >all_align/$lang/$i.align.backward

    ../fast_align/build/atools -c grow-diag-final -i all_align/$lang/$i.align.forward -j all_align/$lang/$i.align.backward > all_align/$lang/$i.align.forward.grow-diag-final
    ../fast_align/build/atools -c invert -i all_align/$lang/$i.align.backward > all_align/$lang/$i.align.backward2
    python3 combine_bidirectional_alignments.py --method grow-diagonal-final all_align/$lang/$i.align.forward all_align/$lang/$i.align.backward2 >all_align/$lang/$i.align.grow-diag-final

  done
done

false && for lang in ende fren enro; do
  lang2=$(echo $lang | sed 's/./& /g' | awk '{print $3, $4, $1, $2}' | sed "s= ==g")
  echo Processing $lang
  mkdir -p all_align/$lang
  base=/home/shuoyangd/projects/realign/${lang}_exps/out/decode/
  for i in 4F9EC56BCB7FD68922D70E08FEBFA1B5 Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none; do
    echo ... Processing $i
    cat $base/$i/out | sed "s= ==g" | sed "s=▁= =g" > all_align/$lang/$i.txt

    paste all_align/$lang/$i.txt data/test/$lang2.lc.tgt | awk -F '\t' '$1!="" && $2!=""' |  sed "s=\t= ||| =g" > all_align/$lang/$i.to_align

    ../fast_align/build/force_align_forward.py  models/$lang2.fw.mdl models/$lang2.fw.err models/$lang2.fw.mdl models/$lang2.fw.err union <all_align/$lang/$i.to_align >all_align/$lang/$i.align.forward
    ../fast_align/build/force_align_backward.py models/$lang2.bw.mdl models/$lang2.bw.err models/$lang2.bw.mdl models/$lang2.bw.err union <all_align/$lang/$i.to_align >all_align/$lang/$i.align.backward

    ../fast_align/build/atools -c grow-diag-final -i all_align/$lang/$i.align.forward -j all_align/$lang/$i.align.backward > all_align/$lang/$i.align.forward.grow-diag-final
    ../fast_align/build/atools -c invert -i all_align/$lang/$i.align.backward > all_align/$lang/$i.align.backward2
    python3 combine_bidirectional_alignments.py --method grow-diagonal-final all_align/$lang/$i.align.forward all_align/$lang/$i.align.backward2 >all_align/$lang/$i.align.grow-diag-final

  done
done

for lang in deen enfr roen ende fren enro; do
  for i in 4F9EC56BCB7FD68922D70E08FEBFA1B5 Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none; do
    python combine.py all_align/$lang/$i.align.forward all_align/$lang/$i.align.backward > all_align/$lang/$i.align.ding
  done
done
