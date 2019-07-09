import "utils/submitters.tape"
import "utils/versioners.tape"

global {

pyenv="/path/to/your/env"
ducttape_output="out"
submitter="shell"

model=(LangPair:
  deen=(Arch:
    transformer="/path/to/mt/system/build/deen_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/path/to/mt/system/build/deen_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/path/to/mt/system/build/deen_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enfr=(Arch:
    transformer="/path/to/mt/system/build/enfr_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/path/to/mt/system/build/enfr_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/path/to/mt/system/build/enfr_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  roen=(Arch:
    transformer="/path/to/mt/system/build/roen_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/path/to/mt/system/build/roen_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/path/to/mt/system/build/roen_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  ende=(Arch:
    transformer="/path/to/mt/system/build/ende_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/path/to/mt/system/build/ende_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/path/to/mt/system/build/ende_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  fren=(Arch:
    transformer="/path/to/mt/system/build/fren_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/path/to/mt/system/build/fren_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/path/to/mt/system/build/fren_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enro=(Arch:
    transformer="/path/to/mt/system/build/enro_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/path/to/mt/system/build/enro_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/path/to/mt/system/build/enro_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
)

saliency=(Saliency: plain guided integral)
normalization=(Normalization: clamp abs)
method=(Method: at sa)  # use attention or saliency
smoothing_factor=(Factor: 0.15 0.0)
nsamples=(NSamples: 30 1 100)

src=(LangPair: deen="de" enfr="en" roen="ro" ende="en" fren="fr" enro="en")
tgt=(LangPair: deen="en" enfr="fr" roen="en" ende="de" fren="en" enro="ro")

flip=(LangPair: deen="" enfr="" roen="" ende="yes" fren="yes" enro="yes")

preprocessed="/home/shuoyangd/projects/realign/alignment-scripts/preprocess/test"
# no need to use different data for different runs
binarized=(LangPair:
  deen="/path/to/mt/system/build/deen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enfr="/path/to/mt/system/build/enfr_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  roen="/path/to/mt/system/build/roen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  ende="/path/to/mt/system/build/ende_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  fren="/path/to/mt/system/build/fren_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enro="/path/to/mt/system/build/enro_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
)

}

package fairseq :: .versioner=disk .path=/home/shuoyangd/projects/realign/fairseq4 { }
package tools :: .versioner=disk .path=/home/shuoyangd/projects/realign/alignment-scripts { }
package fast_align :: .versioner=disk .path=/home/shuoyangd/software/fast_align/build { }

task make_data
  < preprocessed=@
  < binarized=@
  > out
  :: src=@
  :: tgt=@
  :: flip=@ {

  mkdir $out
  if [ -z $flip ] ; then
    ln -s $preprocessed/$src$tgt.lc.src.bpe $out/$src
    ln -s $preprocessed/$src$tgt.lc.tgt.bpe $out/$tgt
    ln -s $preprocessed/$src$tgt.talp $out/ref

    ln -s $binarized/dict.$src.txt $out
    ln -s $binarized/dict.$tgt.txt $out
  else
    ln -s $preprocessed/$tgt$src.lc.tgt.bpe $out/$src
    ln -s $preprocessed/$tgt$src.lc.src.bpe $out/$tgt
    ln -s $preprocessed/$tgt$src.talp $out/ref

    ln -s $binarized/dict.$src.txt $out
    ln -s $binarized/dict.$tgt.txt $out
  fi

}

task align : fairseq
  < data=$out@make_data
  > out
  < model=@
  :: saliency=@
  :: smoothing_factor=@
  :: nsamples=@
  :: normalization=@
  :: src=@
  :: tgt=@
  :: pyenv=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'hostname=b1[123456789]*|c*,mem_free=8g,ram_free=8g,gpu=1'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    sleep `od -An -N1 -i /dev/random` # avoid GPU clash

    if [ $normalization == "clamp" ] ; then
      CUDA_VISIBLE_DEVICES=`free-gpu` python $fairseq/align.py --path $model $data --source-lang $src --target-lang $tgt --saliency $saliency --out $out --smoothing-factor $smoothing_factor --n-samples $nsamples
    elif [ $normalization == "abs" ] ; then
      CUDA_VISIBLE_DEVICES=`free-gpu` python $fairseq/align.py --path $model $data --source-lang $src --target-lang $tgt --saliency $saliency --out $out --abs --smoothing-factor $smoothing_factor --n-samples $nsamples
    fi

    if [ ! -z "`ls $out.*`" ] ; then
      touch $out
    fi
}

task debpe : fairseq
  < in=$out@align
  < data=$out@make_data
  > out
  :: src=@
  :: tgt=@
  :: flip=@
  :: method=@
  :: pyenv=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    if [ -z $flip ] ; then
      python $fairseq/merge_subw_align.py --text $data --alignment $in.$method --source-lang $src --target-lang $tgt --out $out
    else
      python $fairseq/merge_subw_align.py --text $data --alignment $in.$method --source-lang $src --target-lang $tgt --out $out --flip
    fi

    if [ ! -z "`ls $out.*`" ] ; then
      touch $out
    fi
}

task debpe_soft : fairseq
  < in=$out@align
  < data=$out@make_data
  > out
  :: src=@
  :: tgt=@
  :: flip=@
  :: method=@
  :: pyenv=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    if [ -z $flip ] ; then
      python $fairseq/merge_subw_align_soft.py --text $data --alignment $in.$method --source-lang $src --target-lang $tgt --out $out
    else
      python $fairseq/merge_subw_align_soft.py --text $data --alignment $in.$method --source-lang $src --target-lang $tgt --out $out --flip
    fi

    if [ ! -z "`ls $out.*`" ] ; then
      touch $out
    fi
}

task symmetrize : tools fast_align
  < fwd_in=(RealLangPair: deen=$out@debpe[LangPair:deen] enfr=$out@debpe[LangPair:enfr] roen=$out@debpe[LangPair:roen])
  < bwd_in=(RealLangPair: deen=$out@debpe[LangPair:ende] enfr=$out@debpe[LangPair:fren] roen=$out@debpe[LangPair:enro])
  > out
  :: pyenv=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    $fast_align/atools -c invert -i $bwd_in.alg > bwd_in
    $tools/scripts/combine_bidirectional_alignments.py --method grow-diagonal-final $fwd_in.alg bwd_in > $out
}

task aer : tools
  < in=$out@debpe
  < data=$out@make_data
  > aer {

    python $tools/scripts/aer.py $data/ref $in.alg > $aer
}

task aer_bidir : tools
  < in=$out@symmetrize
  < data=(RealLangPair: deen=$out@make_data[LangPair:deen] enfr=$out@make_data[LangPair:enfr] roen=$out@make_data[LangPair:roen])
  > aer {

    python $tools/scripts/aer.py $data/ref $in > $aer
}

plan exp {

  # attention and smoothed attention
  reach aer, aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: at) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (RealLangPair: *)
  reach aer, aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: at) * (Arch: *) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (RealLangPair: *)

  # Ours
  # vanilla
  reach aer, aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: sa) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (RealLangPair: *)
  # SmoothGrad
  reach aer, aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: sa) * (Arch: *) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (RealLangPair: *)

  # Li
  # vanilla
  reach aer, aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: sa) * (Arch: *) * (Normalization: abs) * (Factor: 0.0) * (NSamples: 1) * (RealLangPair: *)
  # SmoothGrad
  reach aer, aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: sa) * (Arch: *) * (Normalization: abs) * (Factor: 0.15) * (NSamples: 30) * (RealLangPair: *)
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}
