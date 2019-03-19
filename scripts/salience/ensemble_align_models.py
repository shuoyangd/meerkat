# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-03-02
#
# Distributed under terms of the MIT license.

import argparse
import logging
import torch
import pdb

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="")
opt_parser.add_argument("--models", nargs='+', default=[], help="")
opt_parser.add_argument("--out", metavar="PATH", type=str, help="")

def main(options):
  alignments = []
  for model_file in options.models:
    alignments.append(torch.load(open(model_file, 'rb'), map_location=lambda storage, loc: storage))  # we are assuming bsz = 1 for this..
  alignments = tuple(alignments)
  ret = []
  for pred in zip(*alignments):
    ens_pred = torch.cat(pred, dim=0)
    ret.append(torch.mean(ens_pred, dim=0).unsqueeze(0))
  torch.save(ret, options.out)

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
