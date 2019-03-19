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
from collections import Counter

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--alignment", metavar="PATH", help="")
opt_parser.add_argument("--out", metavar="PATH", help="")

def main(options):
  alignments = torch.load(open(options.alignment, 'rb'), map_location=lambda storage, loc: storage)  # we are assuming bsz = 1 for this..
  if alignments[0].size(0) != 1:
      alignments = [ torch.mean(alignment, dim=0).unsqueeze(0) for alignment in alignments ]

  alg_out_file = open(options.out + ".alg", 'w')

  for alg in alignments:
    alg = alg.squeeze()  # (tlen, slen)
    alg_out_file.write(str(alg.tolist()) + "\n")

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
