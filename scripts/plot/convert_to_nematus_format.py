# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-07-13
#
# Distributed under terms of the MIT license.

import argparse
import logging
import numpy as np

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="""Converts certain input formats to Nematus alignment dump format.

Input Format:
    hard format: each tok is an alignment point denoted by "src-tgt"
    e.g: 0-0 1-1 2-4 3-2

    soft format: a 2-D list with alignment weights for each point
    e.g.: [[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.02, 0.08, 0.9]]

Output Format:

    each sentence has:

    1. a line showing: <sentence id> ||| <target words> ||| <score> ||| <source words> ||| <number of source words> <number of target words>
    2. then, an array of: <number of target words> + 1 (for <EOS> token) rows with <number of source words> + 1 (for <EOS> token)
    3. and an empty line between sentences

""", formatter_class=argparse.RawDescriptionHelpFormatter)
opt_parser.add_argument("--src", metavar="PATH", required=True, help="")
opt_parser.add_argument("--tgt", metavar="PATH", required=True, help="")
opt_parser.add_argument("--alg", metavar="PATH", required=True, help="")
opt_parser.add_argument("--out", metavar="PATH", default=None, help="")
opt_parser.add_argument("--soft", action='store_true', default=False, help="turn this on if your alignment is soft")


def nan_guard(number, slen):
  if np.isnan(number):
    return 1 / slen
  else:
    return number


def print_array(array):
  array_lines = ""
  slen = len(array[0])
  tlen = len(array)
  for j in range(tlen):
    array_lines += str(nan_guard(array[j][0], slen))
    for i in range(1, slen):
      array_lines += " "
      array_lines += str(nan_guard(array[j][i], slen))
    array_lines += "\n"
  return array_lines


def get_weight(i, j, apairs):
  if (str(i), str(j)) in apairs:
    return 1.0
  else:
    return 0.0


def print_alignment_pairs(apairs, slen, tlen, plus_one=True):
  array_lines = ""
  for j in range(tlen):
    array_lines += str(get_weight(0+1, j+1, apairs))
    for i in range(1, slen):
      array_lines += " "
      array_lines += str(get_weight(i+1, j+1, apairs))
    array_lines += "\n"
  return array_lines


def main(options):
  src_file = open(options.src)
  tgt_file = open(options.tgt)
  alg_file = open(options.alg)
  if options.out:
    out_file = open(options.out, 'w')

  idx = 0
  for sline, tline, aline in zip(src_file, tgt_file, alg_file):
    stoks = sline.strip().split()
    ttoks = tline.strip().split()
    if not options.soft:
      apairs = [ tuple(tok.split('-')) for tok in aline.strip().split() ]
    else:
      aline = aline.replace("nan", "float('nan')")
      aw = eval(aline.strip())

    info_fields = [str(idx)]  # index
    info_fields.append(" ".join(ttoks))  # target token
    info_fields.append("0")  # score -- we don't care, so just give zero
    info_fields.append(" ".join(stoks))  # source token
    info_fields.append("{0} {1}".format(len(stoks), len(ttoks)))  # source len (they assume there is eos), target len
    info_line = " ||| ".join(info_fields)

    # print the alignment weight
    if options.out:
        out_file.write(info_line + "\n")
        if options.soft:
            out_file.write(print_array(aw))
        else:
            out_file.write(print_alignment_pairs(apairs, len(stoks), len(ttoks)))
        out_file.write("\n")
    else:
        sys.stdout.write(info_line + "\n")
        if options.soft:
            sys.stdout.write(print_array(aw))
        else:
            sys.stdout.write(print_alignment_pairs(apairs))
        out_file.write("\n")

    idx += 1

if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
