% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

addpath('/home/fu/workspace/deeplearning/caffe-ssd/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/Deconv_48x48/deploy.prototxt';
modelFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/Deconv_48x48/VGG_VOC0712_SSD_48x48_iter_10000.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

filePre = '/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant162_';
rgbImg = imread([filePre 'rgb.png']);
labels = load([filePre 'box.csv']);

ranInd = randi(size(labels,1));
x1 = round(labels(ranInd,2)*size(rgbImg,2));
y1 = round(labels(ranInd,3)*size(rgbImg,1));
x2 = round(labels(ranInd,4)*size(rgbImg,2));
y2 = round(labels(ranInd,5)*size(rgbImg,1));

% x1 = 60;
% y1 = 60;
% x2 = 100;
% y2 = 100;

img = rgbImg(y1:y2,x1:x2,:);
img = imresize(img,[48,48]);
caffeImg = toCaffeImg(img);
caffeImg(:,:,end+1) = 0;

tic
output = net.forward({caffeImg});
toc

result = output{1};

figure(1);
subplot(2,1,1);
imshow(img);
subplot(2,1,2);
imagesc(result');
% imshow(result'>150);
axis equal;
% priorBoxes = output{2}(:,:,1);
% result_reshape = zeros(size(result,1)*3,5);
% result_reshape(1:3:end,:) = result(:,1:5);
% result_reshape(2:3:end,:) = result(:,6:10);
% result_reshape(3:3:end,:) = result(:,11:15);

% mbox1 = net.blobs('fc7').get_data();
% mbox2 = net.blobs('conv6_mbox_prior').get_data();
% mbox3 = net.blobs('pooling_avg_mbox_prior').get_data();

