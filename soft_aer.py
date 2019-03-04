#!/bin/python

import sys
import math

fw_fname = sys.argv[1]
bw_fname = sys.argv[2]

fw = open(fw_fname, 'r')
bw = open(bw_fname, 'r')

total_kl = 0.0
total_length = 0

for fline in fw:
#  print ("line is", fline)
  fpairs = fline.split()

  bline = bw.readline().replace("nan", "0.0")

  hyp_ali = eval(bline)
#  print (hyp_ali)
#  print (len(hyp_ali), len(hyp_ali[0]))

  ali = {}

  m_src = {}
  m_tgt = {}

  for pairs in fpairs:
    pairs = pairs.replace("p", "-")
        
    a, b = pairs.split('-')
    a = int(a) - 1
    b = int(b) - 1
    a = str(a)
    b = str(b)

    # a is src-word index, b tgt-word index

#    ali[b][a] = 1
    ali[a + " " + b] = 1.0
    m_src[a] = m_src.get(a, 0) + 1.0
    m_tgt[b] = m_tgt.get(b, 0) + 1.0

#  print (ali)
#  print (hyp_ali)

  kl = 0

  for pairs in fpairs:
    pairs = pairs.replace("p", "-")
        
    a, b = pairs.split('-')
    int_a = int(a) - 1
    int_b = int(b) - 1

    a = str(int_a)
    b = str(int_b)

    # a is src-word index, b tgt-word index
#    print(a, b)

    kl += -1.0 / m_tgt[b] * math.log(hyp_ali[int_b][int_a] + 0.0001)

#  print kl, len(ali[0])
  total_kl += kl
  total_length += len(fpairs)

print ("loss = ", total_kl / total_length)


