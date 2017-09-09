## Reimplementation of [HED]() based on official version of caffe

### For training:
1. Download training data from the [original](https://github.com/s9xie/hed) repo, and extract it to $HED/data/;

2. Build caffe with `bash $HED/build.sh`, this will copy reimplemented loss layer to caffe folder first;

3. Download [initial model](http://zhaok-data.oss-cn-shanghai.aliyuncs.com/caffe-model/vgg16convs.caffemodel) and put it
into '$HED/model/';

4. Start to train with `cd $HED && python train.py 2>&1 | tee hed.log`.

### For testing:
1. Download pretrained model from original repo and put it into '$HED/snapshot/';

2. Run `cd $HED && python forward_all()`;

___
By [KAI ZHAO](http://kaiz.xyz)
