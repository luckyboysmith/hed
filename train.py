from __future__ import division
import numpy as np
import sys, os
from os.path import isfile, join
sys.path.insert(0, 'caffe/python')
sys.path.insert(0, 'model')
sys.path.insert(0, 'lib')
import caffe

if not isfile('model/hed_train.pt') or not isfile('model/hed_test.pt') \
or not isfile('model/hed_solver.pt'):
  from hed import make_all
  make_all()

base_weights = 'model/vgg16convs.caffemodel'

caffe.set_mode_gpu()
caffe.set_device(0)
solver = caffe.SGDSolver('model/hed_solver.pt')
solver.net.copy_from(base_weights)
for p in solver.net.params:
  param = solver.net.params[p]
  for i in range(len(param)):
    print p, "param[%d]: mean=%.5f, std=%.5f"%(i, solver.net.params[p][0].data.mean(), \
    solver.net.params[p][0].data.mean())
solver.solve()

