cd /home/fu/workspace/deeplearning/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/Deconv_48x48/solver.prototxt" \
--gpu 0 2>&1 | tee models/VGGNet/VOC0712/Deconv_48x48/VGG_VOC0712_SSD_48x48.log \

