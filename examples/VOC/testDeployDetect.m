% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

addpath('/home/fu/workspace/deeplearning/caffe-ssd/matlab');
load('colors.mat');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

N = 300;

deployFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/VOC/deploy.prototxt';
modelFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/VOC/VOC_SSD_300x300_iter_120000.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

img = imread('/home/fu/workspace/dataset/leftImg8bit/train/cologne/cologne_000009_000019_leftImg8bit.png');

fid = fopen('/media/fu/Elements/tmp/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt');
tline = fgetl(fid);
while ischar(tline)
% Begin of test loop

img = imread(['/media/fu/Elements/tmp/VOCdevkit/VOC2012/JPEGImages/' tline '.jpg']);
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

figure(1);
subplot(2,1,1);
imshow(uint8(img(:,:,1:3)));
hold on;
for maxInd = 1:size(result,2)
    if result(3,maxInd)>0.8
        xy = result(4:7,maxInd);
        xy = xy.*[size(img,2);size(img,1);size(img,2);size(img,1)];
        pw = xy(3)-xy(1);
        ph = xy(4)-xy(2);
        if pw>0 && ph>0
            pcx = (xy(1)+xy(3))*0.5;
            pcy = (xy(2)+xy(4))*0.5;
            rectangle('Position',[xy(1),xy(2),pw,ph],'EdgeColor',colors(result(2,maxInd),:));
        end
    end
end
hold off;


% End of test loop
    disp(tline)
    tline = fgetl(fid);
    pause(1);
end
fclose(fid);