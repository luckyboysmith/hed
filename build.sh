#!/bin/bash
# build caffe source, use "export DEBUG=1" before build will enable debug mode.
set -e
THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $THIS_DIR
cp lib/balance_*.c*   caffe/src/caffe/layers/
cp lib/balance_*layer.hpp  caffe/include/caffe/layers/
cp lib/test_*.cpp  caffe/src/caffe/test/
cd $THIS_DIR/caffe
if [ ! -e build ]; then
    mkdir build
fi
cd build
cmake .. -DUSECUDNN=ON && make -j$(nproc)
cd $THIS_DIR
echo "Done!"

