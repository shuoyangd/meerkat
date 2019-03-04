# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-02-22
#
# Distributed under terms of the MIT license.

import argparse
import logging
import pdb
import torch

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--text", metavar="PATH", help="")
opt_parser.add_argument("--source-lang", help="")
opt_parser.add_argument("--target-lang", help="")
opt_parser.add_argument("--alignment", metavar="PATH", help="")
opt_parser.add_argument("--out", metavar="PATH", help="")
opt_parser.add_argument("--flip", action='store_true', default=False, help="")

def debpe(toks):
  idx_map = {}
  new_idx = -1
  new_toks = []
  for old_idx, tok in enumerate(toks):
    if old_idx == 0 or tok.startswith("▁"):
        new_idx += 1
        idx_map[old_idx] = new_idx
        new_toks.append(tok[1:] if tok.startswith("▁") else tok)
    else:
        idx_map[old_idx] = new_idx
        new_toks[new_idx] += tok
  return idx_map, new_toks

def main(options):
  src_file = open(options.text + "/" + options.source_lang)
  tgt_file = open(options.text + "/" + options.target_lang)
  alignments = torch.load(open(options.alignment, 'rb'), map_location=lambda storage, loc: storage)  # we are assuming bsz = 1 for this..

  src_out_file = open(options.out + ".src", 'w')
  tgt_out_file = open(options.out + ".tgt", 'w')
  alg_out_file = open(options.out + ".alg", 'w')

  for sline, tline, alg in zip(src_file, tgt_file, alignments):
    alg = alg.squeeze()  # (slen, tlen)
    stoks = sline.split()
    ttoks = tline.split()
    stoks = stoks + ["▁<eos>"]
    ttoks = ttoks + ["▁<eos>"]
    _, alg = torch.max(alg, dim=1)  # (tgt_len,)
    smap, debped_stoks = debpe(stoks)
    tmap, debped_ttoks = debpe(ttoks)

    src_out_file.write(" ".join(debped_stoks[:-1]) + "\n")
    tgt_out_file.write(" ".join(debped_ttoks[:-1]) + "\n")
    debpe_alg = []
    old_src_len = len(stoks)
    old_tgt_len = len(ttoks)
    for old_tgt, old_src in enumerate(alg.tolist()):
      if old_tgt != old_tgt_len - 1 and old_src != old_src_len - 1:
        if options.flip:
          debpe_alg.append("{1}-{0}".format(smap[old_src]+1, tmap[old_tgt]+1))
        else:
          debpe_alg.append("{0}-{1}".format(smap[old_src]+1, tmap[old_tgt]+1))
    debpe_alg = set(debpe_alg)
    alg_out_file.write(" ".join(debpe_alg) + "\n")

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
