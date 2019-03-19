import "utils/submitters.tape"
import "utils/versioners.tape"

global {

pyenv="/home/shuoyangd/pyenv/py3t/bin/activate"
ducttape_output="out_v3"
submitter="sge"

#   roen3=(Arch:
#     transformer="/export/c11/shuoyangd/projects/realign/roen_exps_run3/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
#     fconv="/home/shuoyangd/projects/realign/roen_exps_run3/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
#     lstm="/home/shuoyangd/projects/realign/roen_exps_run3/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
#   )
#   roen4=(Arch:
#     transformer="/export/c11/shuoyangd/projects/realign/roen_exps_run4/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
#     fconv="/home/shuoyangd/projects/realign/roen_exps_run4/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
#     lstm="/home/shuoyangd/projects/realign/roen_exps_run4/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
#   )
model=(LangPair:
  deen=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/deen_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/deen_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/deen_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enfr=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/enfr_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/enfr_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/enfr_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  roen=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/roen_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/roen_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/roen_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  deen2=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/deen_exps_run2/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/deen_exps_run2/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/deen_exps_run2/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enfr2=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/enfr_exps_run2/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/enfr_exps_run2/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/enfr_exps_run2/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  roen2=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/roen_exps_run2/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/roen_exps_run2/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/roen_exps_run2/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  deen3=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/deen_exps_run3/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/deen_exps_run3/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/deen_exps_run3/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enfr3=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/enfr_exps_run3/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/enfr_exps_run3/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/enfr_exps_run3/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  roen3=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/roen_exps_run3/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/roen_exps_run3/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/roen_exps_run3/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  deen4=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/deen_exps_run4/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/deen_exps_run4/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/deen_exps_run4/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enfr4=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/enfr_exps_run4/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/enfr_exps_run4/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/enfr_exps_run4/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  roen4=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/roen_exps_run4/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/roen_exps_run4/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/roen_exps_run4/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  ende=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/ende_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/ende_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/ende_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  fren=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/fren_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/fren_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/fren_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
  enro=(Arch:
    transformer="/export/c11/shuoyangd/projects/realign/enro_exps/out/train/4F9EC56BCB7FD68922D70E08FEBFA1B5/out/checkpoint_best.pt"
    fconv="/home/shuoyangd/projects/realign/enro_exps/out/train/Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
    lstm="/home/shuoyangd/projects/realign/enro_exps/out/train/Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out/checkpoint_best.pt"
  )
)

saliency=(Saliency: plain guided)
normalization=(Normalization: clamp abs)
method=(Method: at sa)  # use attention or saliency
smoothing_factor=(Factor: 0.15 0.0)
nsamples=(NSamples: 30 1)

src=(LangPair: deen="de" enfr="en" roen="ro" deen2="de" enfr2="en" roen2="ro" deen3="de" enfr3="en" roen3="ro" deen4="de" enfr4="en" roen4="ro" ende="en" fren="fr" enro="en")
tgt=(LangPair: deen="en" enfr="fr" roen="en" deen2="en" enfr2="fr" roen2="en" deen3="en" enfr3="fr" roen3="en" deen4="en" enfr4="fr" roen4="en" ende="de" fren="en" enro="ro")

flip=(LangPair: deen="" enfr="" roen="" deen2="" enfr2="" roen2="" deen3="" enfr3="" roen3="" deen4="" enfr4="" roen4="" ende="yes" fren="yes" enro="yes")

preprocessed="/home/shuoyangd/projects/realign/alignment-scripts/preprocess/test"
# no need to use different data for different runs
binarized=(LangPair:
  deen="/home/shuoyangd/projects/realign/deen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enfr="/home/shuoyangd/projects/realign/enfr_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  roen="/home/shuoyangd/projects/realign/roen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  deen2="/home/shuoyangd/projects/realign/deen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enfr2="/home/shuoyangd/projects/realign/enfr_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  roen2="/home/shuoyangd/projects/realign/roen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  deen3="/home/shuoyangd/projects/realign/deen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enfr3="/home/shuoyangd/projects/realign/enfr_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  roen3="/home/shuoyangd/projects/realign/roen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  deen4="/home/shuoyangd/projects/realign/deen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enfr4="/home/shuoyangd/projects/realign/enfr_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  roen4="/home/shuoyangd/projects/realign/roen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  ende="/home/shuoyangd/projects/realign/ende_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  fren="/home/shuoyangd/projects/realign/fren_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enro="/home/shuoyangd/projects/realign/enro_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
)

}

package fairseq :: .versioner=disk .path=/home/shuoyangd/projects/realign/fairseq3 { }
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

# ensemble is a bit ugly right now...

task ensemble4_deen : fairseq
  < m1=$out@align[LangPair:deen]
  < m2=$out@align[LangPair:deen2]
  < m3=$out@align[LangPair:deen3]
  < m4=$out@align[LangPair:deen4]
  > out
  :: pyenv=@
  :: method=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    python $fairseq/ensemble_align_models.py --models $m1.$method $m2.$method $m3.$method $m4.$method --out $out
    if [ ! -z "ls $out" ] ; then
      touch $out
    fi
}

task ensemble4_enfr : fairseq
  < m1=$out@align[LangPair:enfr]
  < m2=$out@align[LangPair:enfr2]
  < m3=$out@align[LangPair:enfr3]
  < m4=$out@align[LangPair:enfr4]
  > out
  :: pyenv=@
  :: method=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    python $fairseq/ensemble_align_models.py --models $m1.$method $m2.$method $m3.$method $m4.$method --out $out
    if [ ! -z "ls $out" ] ; then
      touch $out
    fi
}

task ensemble4_roen : fairseq
  < m1=$out@align[LangPair:roen]
  < m2=$out@align[LangPair:roen2]
  < m3=$out@align[LangPair:roen3]
  < m4=$out@align[LangPair:roen4]
  > out
  :: pyenv=@
  :: method=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    python $fairseq/ensemble_align_models.py --models $m1.$method $m2.$method $m3.$method $m4.$method --out $out
    if [ ! -z "ls $out" ] ; then
      touch $out
    fi
}

# end of ensemble tasks

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

task debpe_ensemble : fairseq
  < in=(RealLangPair: deen=$out@ensemble4_deen enfr=$out@ensemble4_enfr roen=$out@ensemble4_roen)
  < data=(RealLangPair: deen=$out@make_data[LangPair:deen] enfr=$out@make_data[LangPair:enfr] roen=$out@make_data[LangPair:roen])
  > out
  :: src=(RealLangPair: deen="de" enfr="en" roen="ro")
  :: tgt=(RealLangPair: deen="en" enfr="fr" roen="en")
  :: method=@
  :: pyenv=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    python $fairseq/merge_subw_align.py --text $data --alignment $in --source-lang $src --target-lang $tgt --out $out

    if [ ! -z "`ls $out.*`" ] ; then
      touch $out
    fi
}

task debpe_ensemble_soft : fairseq
  < in=(RealLangPair: deen=$out@ensemble4_deen enfr=$out@ensemble4_enfr roen=$out@ensemble4_roen)
  < data=(RealLangPair: deen=$out@make_data[LangPair:deen] enfr=$out@make_data[LangPair:enfr] roen=$out@make_data[LangPair:roen])
  > out
  :: src=(RealLangPair: deen="de" enfr="en" roen="ro")
  :: tgt=(RealLangPair: deen="en" enfr="fr" roen="en")
  :: method=@
  :: pyenv=@
  :: .submitter=$submitter
  :: .resource_flags="-l 'mem_free=16g,ram_free=16g'"
  :: .action_flags="-m ae -M dings@jhu.edu" {

    python $fairseq/merge_subw_align_soft.py --text $data --alignment $in --source-lang $src --target-lang $tgt --out $out

    if [ ! -z "`ls $out.*`" ] ; then
      touch $out
    fi
}

task avg_aer : tools
  < aer1=(RealLangPair: deen=$aer@aer[LangPair:deen] enfr=$aer@aer[LangPair:enfr] roen=$aer@aer[LangPair:roen])
  < aer2=(RealLangPair: deen=$aer@aer[LangPair:deen2] enfr=$aer@aer[LangPair:enfr2] roen=$aer@aer[LangPair:roen2])
  < aer3=(RealLangPair: deen=$aer@aer[LangPair:deen3] enfr=$aer@aer[LangPair:enfr3] roen=$aer@aer[LangPair:roen3])
  < aer4=(RealLangPair: deen=$aer@aer[LangPair:deen4] enfr=$aer@aer[LangPair:enfr4] roen=$aer@aer[LangPair:roen4])
  > aer_mean
  > aer_std
  :: pyenv=@
  {

    aer1=`cat $aer1 | grep -Eo "[0-9\.]+\%" | grep -Eo "[0-9\.]+" | head -1`
    aer2=`cat $aer2 | grep -Eo "[0-9\.]+\%" | grep -Eo "[0-9\.]+" | head -1`
    aer3=`cat $aer3 | grep -Eo "[0-9\.]+\%" | grep -Eo "[0-9\.]+" | head -1`
    aer4=`cat $aer4 | grep -Eo "[0-9\.]+\%" | grep -Eo "[0-9\.]+" | head -1`
    echo $PWD > aer_mean
    python -c "import numpy as np; l = [$aer1, $aer2, $aer3, $aer4]; print(np.mean(l))" >> aer_mean
    echo $PWD > aer_std
    python -c "import numpy as np; l = [$aer1, $aer2, $aer3, $aer4]; print(np.std(l))" >> aer_std
}

task symmetrize : tools fast_align
  < fwd_in=$out@debpe_ensemble
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
  < in=(IsEnsemble: no=$out@debpe yes=$out@debpe_ensemble)
  < data=(IsEnsemble:
      no=$out@make_data
      yes=(RealLangPair: deen=$out@make_data[LangPair:deen] enfr=$out@make_data[LangPair:enfr] roen=$out@make_data[LangPair:roen])
    )
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
  # reach aer via (LangPair: *) * (Saliency: *) * (Method: *) * (Arch: *)
  reach avg_aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: transformer) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: no) * (RealLangPair: *)
  reach aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: transformer) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: yes)
  reach avg_aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: fconv) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: no) * (RealLangPair: *)
  reach aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: fconv) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: yes)
  reach avg_aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: lstm) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: no) * (RealLangPair: *)
  reach aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: lstm) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: yes)
  reach avg_aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (IsEnsemble: no) * (RealLangPair: *)
  reach aer via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (IsEnsemble: yes)

  reach debpe_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: transformer) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: no) * (RealLangPair: *)
  reach debpe_ensemble_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: transformer) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: yes)
  reach debpe_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: fconv) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: no) * (RealLangPair: *)
  reach debpe_ensemble_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: fconv) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: yes)
  reach debpe_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: lstm) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: no) * (RealLangPair: *)
  reach debpe_ensemble_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: lstm) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (IsEnsemble: yes)
  reach debpe_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (IsEnsemble: no) * (RealLangPair: *)
  reach debpe_ensemble_soft via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (IsEnsemble: yes)

  reach aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: transformer) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (RealLangPair: *)
  reach aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: fconv) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (RealLangPair: *)
  reach aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: lstm) * (Normalization: clamp) * (Factor: 0.15) * (NSamples: 30) * (RealLangPair: *)
  reach aer_bidir via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0) * (NSamples: 1) * (RealLangPair: *)
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}
