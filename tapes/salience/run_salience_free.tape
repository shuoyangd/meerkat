import "utils/submitters.tape"
import "utils/versioners.tape"

global {

pyenv="/path/to/your/env"
ducttape_output="out_free"
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

saliency=(Saliency: plain guided)
normalization=(Normalization: clamp abs)
method=(Method: at sa)  # use attention or saliency
smoothing_factor=(Factor: 0.0 0.15)
nsamples=(NSamples: 1 30)

src=(LangPair: deen="de" enfr="en" roen="ro" ende="en" fren="fr" enro="en")
tgt=(LangPair: deen="en" enfr="fr" roen="en" ende="de" fren="en" enro="ro")

flip=(LangPair: deen="" enfr="" roen="" ende="yes" fren="yes" enro="yes")

preprocessed="/path/to/alignment-scripts/preprocess/test"
# no need to use different data for different runs
binarized=(LangPair:
  deen="/path/to/mt/system/build/deen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enfr="/path/to/mt/system/build/enfr_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  roen="/path/to/mt/system/build/roen_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  ende="/path/to/mt/system/build/ende_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  fren="/path/to/mt/system/build/fren_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
  enro="/path/to/mt/system/build/enro_exps/out/binarize_data/DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none/out"
)

silver_data_dir="/path/to/silver_data"
arch_branch_name=(Arch:
  transformer="4F9EC56BCB7FD68922D70E08FEBFA1B5"
  fconv="Architecture.fconv_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none"
  lstm="Architecture.lstm_wiseman_iwslt_de_en+ClipNorm.0.1+DoTokenize.no+DoTruecase.no+MergeTest.yes+SubwordMethod.none"
)

}

package fairseq :: .versioner=git .repo="https://github.com/shuoyangd/fairseq" .ref=saliency { }
package tools :: .versioner=git .repo="https://github.com/shuoyangd/tape4nmt-tools" .ref=HEAD { }
package fast_align :: .versioner=disk .path="/path/to/built/fast-align" { }

task make_data
  < preprocessed=@
  < binarized=@
  > out
  :: src=@
  :: tgt=@
  :: silver_data_dir=@
  :: arch_branch_name=@
  :: flip=@ {

  mkdir $out
  if [ -z $flip ] ; then
    ln -s $preprocessed/$src$tgt.lc.src.bpe $out/$src
    ln -s $binarized/dict.$src.txt $out
    ln -s $binarized/dict.$tgt.txt $out
  else
    ln -s $preprocessed/$tgt$src.lc.tgt.bpe $out/$src
    ln -s $binarized/dict.$src.txt $out
    ln -s $binarized/dict.$tgt.txt $out
  fi

  # ln -s $silver_data_dir/$src$tgt/tgt/$arch_branch_name.txt $out/$tgt
  # ln -s $silver_data_dir/$src$tgt/ref/$arch_branch_name.align.ding $out/ref
  ln -s $silver_data_dir/$src$tgt/ref/$arch_branch_name.align.ding $out/ref
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
      python $fairseq/merge_subw_align.py --text $data --alignment $in.$method --source-lang $src --target-lang $tgt --out $out --dont-add-one
    else
      python $fairseq/merge_subw_align.py --text $data --alignment $in.$method --source-lang $src --target-lang $tgt --out $out --dont-add-one --flip
    fi

    if [ ! -z "`ls $out.*`" ] ; then
      touch $out
    fi
}

task aer : tools
  < in=$out@debpe
  < data=$out@make_data
  > aer {

    python $tools/scripts/aer.py $data/ref $in.alg > $aer
}

plan exp {
  # reach aer via (LangPair: *) * (Saliency: *) * (Method: *) * (Arch: *)
  reach aer,align via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0 0.15) * (NSamples: 1 30)
  reach aer,align via (LangPair: *) * (Saliency: plain) * (Method: *) * (Arch: *) * (Normalization: clamp) * (Factor: 0.0 0.15) * (NSamples: 1 30)
}

global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

