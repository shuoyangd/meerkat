#!/bin/python

import sys
import math
import torch
import pdb

file_name = sys.argv[1]

def plogp(p, dim = -1, keepdim = None):
     return -torch.where(p > 0, p * p.log(), p.new([0.0]))
#     return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim, keepdim = keepdim) # can be a scalar, when PyTorch.supports it

l = torch.load(open(file_name, 'rb'), map_location=lambda storage, loc: storage)

# tmp = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
# print (tmp)
# print (plogp(tmp))
# 
# tmp = torch.tensor([[0.5, 0.5], [0.3, 0.7]])
# print (tmp)
# print (plogp(tmp))

a = 0.0
b = 0.0

for i in l:
#  print (i.tolist())
  sample, tgt, src = len(i), len(i[0]), len(i[0][0])
  i = i.mean(0).view(tgt, src)

  a += (plogp(i).sum(dim=1).mean().tolist())
  b += 1
#  print (plogp(i).sum(dim=1).mean().tolist())
#  break
#
#  i = i.tolist()
#  print (i)
print (a / b)
