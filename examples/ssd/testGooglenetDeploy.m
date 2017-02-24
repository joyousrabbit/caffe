addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = 'ssd_googlenet_deploy.prototxt';
modelFile = '/media/fu/Elements/nul/ssd_googlenet_iter_1000000.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

img = zeros(160,160)+200;
img = fillCircle(img,80,110,30,100);
img = fillCircle(img,80,80,30,50);
% img = fillRectangle(img,50,50,60,60,200);
caffeImg = toCaffeImg(img);

tic
output = net.forward({caffeImg});
toc

result = output{1};
priorBoxes = output{2}(:,:,1);
result_reshape = zeros(size(result,1)*3,5);
result_reshape(1:3:end,:) = result(:,1:5);
result_reshape(2:3:end,:) = result(:,6:10);
result_reshape(3:3:end,:) = result(:,11:15);

% mbox1 = net.blobs('5b_mbox_prior').get_data();
% mbox2 = net.blobs('conv6_mbox_prior').get_data();
% mbox3 = net.blobs('pooling_avg_mbox_prior').get_data();

figure(1);
imshow(uint8(img));
N = 160;
hold on;
inds = find(result_reshape(:,5)>1.4);
for maxInd = inds'
% [maxV,maxInd] = max(result_reshape(:,5));
priorBox = priorBoxes(:,maxInd);
corri = result_reshape(maxInd,:);
xy = priorBox*N;
pw = xy(3)-xy(1);
ph = xy(4)-xy(2);
pcx = (xy(1)+xy(3))*0.5;
pcy = (xy(2)+xy(4))*0.5;
rectangle('Position',[xy(1),xy(2),pw,ph],'EdgeColor','red');
cx = corri(1)*pw+pcx;
cy = corri(2)*ph+pcy;
w = pw*exp(corri(3));
h = ph*exp(corri(4));
rectangle('Position',[cx-0.5*w,cy-0.5*w,w,h],'EdgeColor','green');
end
hold off;
