#!/usr/bin/python

import sys

# print 'Number of arguments:', len(sys.argv), 'arguments.'
# print 'Argument List:', str(sys.argv)

fw_fname = sys.argv[1]
bw_fname = sys.argv[2]

fw = open(fw_fname, 'r')
bw = open(bw_fname, 'r')

for fline in fw:
  bline = bw.readline()

#  fline = fline.strip()
#  bline = bline.strip()
  fpairs = fline.split()
  bpairs = bline.split()

  m = {}
  for i in fpairs:
    m[i] = m.get(i, 0) + 1
  for i in bpairs:
    m[i] = m.get(i, 0) + 1

  ans = ""
  for k,v in m.items():
    if v == 2:
      ans += k + " "
    else:
      a, b = k.split('-')
      ans += a+'p'+b + " "
  print ans
