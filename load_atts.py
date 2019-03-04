#!/bin/python

import sys
import math
import torch
import pdb

file_name = sys.argv[1]

l = torch.load(open(file_name, 'rb'), map_location=lambda storage, loc: storage)

for i in l:
#  print (i.tolist())
  sample, tgt, src = len(i), len(i[0]), len(i[0][0])
  i = i.mean(0).view(tgt, src)
  i = i.tolist()
  print (i)
