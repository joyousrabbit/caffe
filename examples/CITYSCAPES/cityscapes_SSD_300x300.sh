cd /home/fu/workspace/deeplearning/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/CITYSCAPE/solver.prototxt" \
--gpu 0 2>&1 | tee models/VGGNet/VOC0712/CITYSCAPE/train.log \

