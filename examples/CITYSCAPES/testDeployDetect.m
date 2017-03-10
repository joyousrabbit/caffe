% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

addpath('/home/fu/workspace/deeplearning/caffe-ssd/matlab');
load('colors.mat');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

N = 300;

deployFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/CITYSCAPE/deploy.prototxt';
modelFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/CITYSCAPE/CITYSCAPE_SSD_300x300_iter_20000.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/munich/munich_000047_000019_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/bonn/bonn_000045_000019_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_005403_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_005403_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_005403_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_005403_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_005403_leftImg8bit.png');
img = imread('/home/fu/workspace/dataset/leftImg8bit/train/cologne/cologne_000009_000019_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_008001_leftImg8bit.png');
% img = imread('/home/fu/workspace/dataset/leftImg8bit/test/mainz/mainz_000000_005549_leftImg8bit.png');


% filePath = '/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/A1_training/plant159_';
% img = imread([filePath 'rgb.png']);
img = imresize(img,[N,N]);
% figure(1);
% imshow(uint8(img));

% label = imread([filePath 'label.png']); 
% inds = unique(label(:));
% inds(inds==0) = [];
% detected = inds(rand(1,length(inds))>0.5);
% bwLabel = ismember(label,double(detected));
bwLabel = 0;

caffeImg = toCaffeImg(img);
% bwLabel = imresize(bwLabel,[N,2*N]);
figure(1);subplot(2,1,2);imshow(bwLabel);
caffeImg(:,:,end+1) = bwLabel'*255;

tic
output = net.forward({caffeImg});
toc

result = output{1};
% priorBoxes = output{2}(:,:,1);
% result_reshape = zeros(size(result,1)*3,5);
% result_reshape(1:3:end,:) = result(:,1:5);
% result_reshape(2:3:end,:) = result(:,6:10);
% result_reshape(3:3:end,:) = result(:,11:15);

% mbox1 = net.blobs('fc7').get_data();
% mbox2 = net.blobs('conv6_mbox_prior').get_data();
% mbox3 = net.blobs('pooling_avg_mbox_prior').get_data();

figure(1);
subplot(2,1,1);
imshow(uint8(img(:,:,1:3)));
hold on;
for maxInd = 1:size(result,2)
    if result(3,maxInd)>0.2
        xy = result(4:7,maxInd);
        xy = xy.*[size(img,2);size(img,1);size(img,2);size(img,1)];
        pw = xy(3)-xy(1);
        ph = xy(4)-xy(2);
        pcx = (xy(1)+xy(3))*0.5;
        pcy = (xy(2)+xy(4))*0.5;
        rectangle('Position',[xy(1),xy(2),pw,ph],'EdgeColor',colors(result(2,maxInd),:));
    end
end
hold off;
