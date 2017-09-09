# -*- coding: utf-8 -*-
import numpy as np
import scipy.misc
import Image
import scipy.io
import os
from os.path import join
import sys
sys.path.insert(0, 'caffe/python')
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('model/hed_test.pt','snapshot/hed_pretrained_bsds.caffemodel', caffe.TEST)
test_dir = 'data/HED-BSDS/test' # test images directory
save_dir = 'data/edge-results/hed_pretrained' # directory to save results
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
imgs = [i for i in os.listdir(test_dir) if '.jpg' in i]
nimgs = len(imgs)
print "totally "+str(nimgs)+"images"
for i in range(nimgs):
  img = imgs[i]
  img = Image.open(join(test_dir, img))
  img = np.array(img, dtype=np.float32)
  img = img[:,:,::-1]
  img -= np.array((104.00698793,116.66876762,122.67891434))
  img = img.transpose((2,0,1))
  net.blobs['data'].reshape(1, *img.shape)
  net.blobs['data'].data[...] = img
  net.forward()
  fuse = net.blobs['sigmoid_fuse'].data[0][0,:,:]
  scipy.io.savemat(join(save_dir, imgs[i][0:-4]),dict({'sk':1-fuse/fuse.max()}),appendmat=True)
  scipy.misc.imsave(join(save_dir, imgs[i]),1-fuse/fuse.max())
  print "Saving to '" + join(save_dir, imgs[i][0:-4]) + "', Processing %d of %d..."%(i + 1, nimgs)

