#!/bin/bash

import sys
import math

fw_fname = sys.argv[1]
bw_fname = sys.argv[2]

fw = open(fw_fname, 'r')
bw = open(bw_fname, 'r')

total_kl = 0.0
total_length = 0

for fline in fw:
  fpairs = fline.split()

  bline = bw.readline()
  hyp_ali = eval(bline)

  ali = eval(bline) # make a copy
  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      ali[i][j] = 0

  m_src = {}
  m_tgt = {}
  for pairs in fpairs:
    a, b = pairs.split('-')
    a = int(a)
    b = int(b)

    m_src[a] = m_src.get(a, 0) + 1
    m_tgt[b] = m_tgt.get(b, 0) + 1

    ali[a][b] = 1

  kl = 0
  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      ali[i][j] /= m_tgt[j]

  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      if ali[i][j] != 0:
        kl += -ali[i][j] * math.log(hyp_ali[i][j])

  print kl, len(ali[0])
  total_kl += kl
  total_length += len(ali[0])

print "loss = ", total_kl / total_length


