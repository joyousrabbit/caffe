addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = 'ssd_googlenet_deploy.prototxt';
% modelFile = '/media/fu/Elements/nul/blackpoint_googlenet_iter_90345.caffemodel';
% net = caffe.Net(deployFile,modelFile,'test');
net = caffe.Net(deployFile,'test');

img = zeros(160,160)+0;
img = fillCircle(img,100,50,20,210);
img = fillCircle(img,40,40,16,200);
% img = fillRectangle(img,50,50,60,60,200);
caffeImg = toCaffeImg(img);

tic
output = net.forward({caffeImg});
toc

mbox1 = net.blobs('5b_mbox_prior').get_data();
mbox2 = net.blobs('conv6_mbox_prior').get_data();
mbox3 = net.blobs('pooling_avg_mbox_prior').get_data();

for i = 1:size(mbox1,2)
figure(1);
imshow(uint8(img));
N = 160;
hold on;

% for i = 1:3
    x1 = mbox1(1,i,1);
    y1 = mbox1(2,i,1);
    x2 = mbox1(3,i,1);
    y2 = mbox1(4,i,1);
    rectangle('Position',[x1,y1,x2-x1,y2-y1]*N,'EdgeColor','green');

hold off;
pause(0.5)

end
