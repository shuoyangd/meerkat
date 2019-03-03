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

  ali = eval(bline) # make a copy
#  print (len(ali), len(ali[0]))
  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      ali[i][j] = 0.00000001

  m_src = {}
  m_tgt = {}
  for pairs in fpairs:
    pairs = pairs.replace("p", "-")
        
    a, b = pairs.split('-')
    a = int(a) - 1
    b = int(b) - 1

    ali[b][a] = 1


  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      m_src[j] = m_src.get(j, 0) + ali[i][j]
      m_tgt[i] = m_tgt.get(i, 0) + ali[i][j]


  ali[len(ali) - 1][len(ali[0]) - 1] = 1 # for EOS -EOS alignment
  m_src[len(ali[0]) - 1] = 1
  m_tgt[len(ali) - 1] = 1

#  print (ali)
#  print (hyp_ali)

  kl = 0
  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      ali[i][j] /= m_tgt[i]


  for i in range(0, len(ali)):
    for j in range(0, len(ali[i])):
      if ali[i][j] != 0:
#        print (hyp_ali[i][j])
        kl += -ali[i][j] * math.log(hyp_ali[i][j] + 0.0001)

#  print kl, len(ali[0])
  total_kl += kl
  total_length += len(ali[0]) + 1

print ("loss = ", total_kl / total_length)


