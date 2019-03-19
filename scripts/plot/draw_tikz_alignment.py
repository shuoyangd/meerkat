# -*- coding: utf-8 -*-
#
# Copyright © 2019 Shuoyang Ding <shuoyangd@gmail.com>
# Created on 2019-02-23
#
# Distributed under terms of the MIT license.

import argparse
import logging
import numpy as np
import sys

logging.basicConfig(
  format='%(asctime)s %(levelname)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

opt_parser = argparse.ArgumentParser(description="""Generates tikz code that plots alignment.

hard format: each tok is an alignment point denoted by "src-tgt"
e.g: 0-0 1-1 2-4 3-2

soft format: a 2-D list with alignment weights for each point
[[0.5, 0.3, 0.2], [0.1, 0.8, 0.1], [0.02, 0.08, 0.9]]
""")
opt_parser.add_argument("--src", metavar="PATH", required=True, help="")
opt_parser.add_argument("--tgt", metavar="PATH", required=True, help="")
opt_parser.add_argument("--alg", metavar="PATH", required=True, help="")
opt_parser.add_argument("--out", metavar="PATH", default=None, help="")
opt_parser.add_argument("--soft", action='store_true', default=False, help="turn this on if your alignment is soft")

template="""% https://gist.github.com/tetsuok/2979324
\\begin{{tikzpicture}}[scale=0.7]
  \\draw[step=1cm,draw=black,line width=1.5] (0,0) grid ({0},{1});

  \\foreach \\y/\\f in {{ {2} }} {{
    \\node[font=\\sffamily,left] at (-.2,{1}.5-1-\\y) {{{{\\raggedleft \\f }}}};
  }}

  \\foreach \\x/\\e in {{ {3} }} {{
    \\node[font=\\sffamily,rotate=60,right] at (\\x+.4,{1}.2) {{{{\\raggedright \\e}}}};
  }}

  % draw word alignment
  {4}
\end{{tikzpicture}}
"""

def escape_tikz_reserves(s):
  s = s.replace(",", "{,}")
  s = s.replace(".", "{.}")
  s = s.replace("-", "{-}")
  s = s.replace("+", "{+}")
  s = s.replace("▁", "\_")
  return s

def nan_guard(number, slen):
  if np.isnan(number):
    return 1 / slen
  else:
    return number

def main(options):
  src_file = open(options.src)
  tgt_file = open(options.tgt)
  alg_file = open(options.alg)
  if options.out:
    out_file = open(options.out, 'w')
  for sline, tline, aline in zip(src_file, tgt_file, alg_file):
    stoks = [ escape_tikz_reserves(tok) for tok in sline.strip().split() ]
    ttoks = [ escape_tikz_reserves(tok) for tok in tline.strip().split() ]
    if not options.soft:
      apairs = [ tuple(tok.split('-')) for tok in aline.strip().split() ]
    else:
      aline = aline.replace("nan", "float('nan')")
      aw = eval(aline.strip())

    fields = []
    fields.append(len(ttoks))  # {0}
    fields.append(len(stoks))  # {1}
    fields.append(", ".join(["{0}/{1}".format(idx, tok) for idx, tok in enumerate(stoks)]))  # {2}
    fields.append(", ".join(["{0}/{1}".format(idx, tok) for idx, tok in enumerate(ttoks)]))  # {3}

    # hard
    if not options.soft:
      plot_vertex = "\\foreach \\x/\\y in { " + \
        ", ".join(["{0}/{1}".format(int(tup[1])-1, int(tup[0])-1) for tup in apairs]) + \
        " } {\n"
      plot_vertex += "    \\node[rectangle,fill=black!75!white,draw=black!75!white,minimum size=0.6cm,line width=2] at (\\x+.5, {0}.5-\\y) {{}};\n".format(len(stoks)-1)
      plot_vertex += "  }\n"
    # soft
    else:
      plot_vertex=""
      for i in range(len(stoks)):
        for j in range(len(ttoks)):
            plot_vertex += "  \\node[rectangle,fill=black!{0}!white,draw=black!{1}!white,minimum size=0.6cm,line width=2] at ({3}, {2}) {{}};\n".format(\
                nan_guard(aw[j][i] * 150, len(stoks)), \
                nan_guard(aw[j][i] * 150, len(stoks)), \
                len(stoks)-0.5-i, j+0.5)
    fields.append(plot_vertex)  # {4}

    if options.out:
      out_file.write(template.format(*tuple(fields)))
    else:
      sys.stdout.write(template.format(*tuple(fields)))


if __name__ == "__main__":
  ret = opt_parser.parse_known_args()
  options = ret[0]
  if ret[1]:
    logging.warning(
      "unknown arguments: {0}".format(
      opt_parser.parse_known_args()[1]))

  main(options)
