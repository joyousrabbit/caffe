cd /home/fu/workspace/deeplearning/caffe-ssd
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_300x300_googlenet_CVPPP/solver.prototxt" \
--snapshot="models/VGGNet/VOC0712/SSD_300x300_googlenet_CVPPP/VGG_VOC0712_SSD_300x300_iter_40000.solverstate" \
--gpu 0 2>&1 | tee models/VGGNet/VOC0712/SSD_300x300_googlenet_CVPPP/VGG_VOC0712_SSD_300x300.log \

