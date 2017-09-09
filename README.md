## Reimplementation of [HED](https://github.com/s9xie/hed) based on official version of caffe

### For training:
1. Clone this code by `git clone https://github.com/zeakey/hed --recursive`, assume your source code directory is`$HED`;

2. Download [training data](http://vcl.ucsd.edu/hed/HED-BSDS.tar) from the [original](https://github.com/s9xie/hed) repo, and extract it to `$HED/data/`;

3. Build caffe with `bash $HED/build.sh`, this will copy reimplemented loss layer to caffe folder first;

4. Download [initial model](http://zhaok-data.oss-cn-shanghai.aliyuncs.com/caffe-model/vgg16convs.caffemodel) and put it
into `$HED/model/`;

5. Start to train with `cd $HED && python train.py 2>&1 | tee hed.log`(will generate training/testing network prototxt
automatically by calling `$HED/model/hed.py`).

### For testing:
1. Download [pretrained model](http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel) from original repo and put it into `$HED/snapshot/`;

2. Generate testing network prototxt by `python $HED/model/hed.py`(will generate training network prototxt as well); 

3. Run `cd $HED && python forward_all()`;

___
By [KAI ZHAO](http://kaiz.xyz)
