# meerkat

This repository will allow you to reproduce the results in our WMT 2019 paper [Saliency-driven Word Alignment Interpretation for Neural Machine Translation](https://arxiv.org/pdf/1906.10282.pdf).

As most research paper nowadays, the pipeline for the experiments described in our paper is awfully long.
Hence, to foster easy and reliable reproduction of results, we'll be heavily relying on [ducttape](https://github.com/jhclark/ducttape).

### What is this ducttape thing?

ducttape is a Linux experimental management system created by the wonderful [Jonathan Clark](http://www.cs.cmu.edu/~jhclark/) who used to be a PhD student in NLP himself.
It's supposed to help creating replicable and manageable pipelines for academic researchers working on Linux.

Setting up is pretty easy. You can either [download the tarball I built](http://cs.jhu.edu/~sding/downloads/ducttape.tar) or follow their [readme](https://github.com/jhclark/ducttape) to build your own.
If you choose to use my tarball, you'll get a jar `ducttape.jar` and an executable script `ducttape` upon untarring.
If you are able to run the `ducttape` script, you are good to go.

### Prepare Data

Special thanks to Thomas Zenkel, Joern Wuebker and John DeNero, authors of [this paper](https://arxiv.org/abs/1901.11359). They definitely made this process much less painful than it usually is.

I've made their experiment script a submodule (`alignment-scripts`).
Just navigate into that directory and follow their instruction to preprocess the data.

### Build MT System (Optional)

Our experiments involves building several machine translation models.
You can choose to [download the model kit](http://cs.jhu.edu/~sding/downloads/model_kit.tgz) we prepared, or build your own.
You can skip this section if you use the model kit.

If you choose to reproduce the system as well, follow the steps below:

1. Setup [tape4nmt](https://github.com/shuoyangd/tape4nmt), the ducttape workflow I use for building NMT systems.
2. Checkout this repo (we'll be referring the directory as `/path/to/repo` below). Navigate to `/path/to/repo/tapes/mt`. Here, the `*.tape` files specify the pipelines, and `*.tconf` files specify the configuration/hyperparameters. You'll need to update some configurations in `*.tconf` files. The `*.tconf` files are supposed to be self-explanatory.
3. Copy all the files in that folder to the `tape4nmt` directory.
4. Within the `tape4nmt` directory, run the following bash command to build systems:

```bash
# deen
ducttape de-en-de.tape -C deen.tconf
# ende
ducttape de-en-de.tape -C ende.tconf
# enfr
ducttape en-fr-en.tape -C enfr.tconf
# fren
ducttape en-fr-en.tape -C fren.tconf
# roen
ducttape ro-en-ro.tape -C roen.tconf
# enro
ducttape ro-en-ro.tape -C enro.tconf
```

That's it! If things work out correctly, you should get exactly the same model as I did.

### Reproduce Numbers

By now, you should have either [downloaded the model kit](http://cs.jhu.edu/~sding/downloads/model_kit.tgz) or built your system and obtained the decoder output.

1. If you haven't yet, checkout this repo (we'll be referring the directory as `/path/to/repo` below). Navigate to `/path/to/repo/tapes/salience`. You should see two files with suffix `*.tape`, where `run_salience.tape` and `run_salience_free.tape` will allow you to reproduce Table 2 and Table 3 in the paper, respectively.
2. Update some configurations in `*.tape` files. They are supposed to be self-explanatory.
3. Within the `/path/to/repo/tapes/salience` directory, run the following bash command to reproduce experiments:

```bash
# reproduce table 2
ducttape run_salience.tape
# reproduce table 3
ducttape run_salience_free.tape
```

That's it! You should get roughly same numbers. It's not going to be exactly the same, due to the randomness involved in SmoothGrad.

### Misc

You can find some scripts we used for analysis and some sanity checks in `scripts/analysis`.
They are not supposed to be clean enough to run out-of-the-box, but only to provide reference if you are interested in reproducing them as well.

+ reproducing all fast-align results, including online results: `scripts/analysis/run_all_align.sh`
+ dispersion: `scripts/analysis/run_entropy.sh`

I used `scripts/plot/draw_tikz_alignment.py` to draw the figures in the paper.

### I'd like to understand and re-use this codebase. Where do I start?

First of all, the codebase for this paper involves lots of deeply-coupled changes on top of the [fairseq](https://github.com/pytorch/fairseq) toolkit, which is not the best way to do it (talk to me if you need to migrate this to other things you are interested in).

If you just want to understand the implementation for word alignment interpretations, the entry point for that part of the code is [align.py](https://github.com/shuoyangd/fairseq/blob/ff3eaf96639fc077686aa01f889f6253f6012cd3/align.py).
At a very high-level, here is how it is done: for the embedding of each input words, I add a backward hook which asks them to log their gradient during back-propagation into a singleton object called `SaliencyManager` (defined in [`fairseq_model.py`](https://github.com/shuoyangd/fairseq/blob/ff3eaf96639fc077686aa01f889f6253f6012cd3/fairseq/models/fairseq_model.py)).
I then retrieve the logged gradients from `SaliencyManager` to calculate the saliency score for each word.

### Citation

```
@inproceedings{ding-etal-2019-saliency,
    title = "Saliency-driven Word Alignment Interpretation for Neural Machine Translation",
    author = "Ding, Shuoyang  and
      Xu, Hainan  and
      Koehn, Philipp",
    booktitle = "Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers)",
    month = aug,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-5201",
    doi = "10.18653/v1/W19-5201",
    pages = "1--12",
}
```

### Naming

Meerkats are small canivores living in all parts of the Kalahari Desert in Botswana, in much of the Namib Desert in Namibia and southwestern Angola, and in South Africa (from [wikipedia](https://en.wikipedia.org/wiki/Meerkat)). Meerkats are very social animals, as they tend to live in clans. It is common to see clans of meerkats standing **aligned**.
