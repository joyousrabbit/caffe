addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = 'blackpoint_googlenet_deploy.prototxt';
modelFile = '/media/fu/Elements/nul/blackpoint_vgg_iter_275821.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

img = zeros(160,160);
img = fillCircle(img,80,100,10,255);
caffeImg = toCaffeImg(img);

tic
output = net.forward({caffeImg});
toc

% w1 = net.layers('conv4_stage1').params(1).get_data();
% w2 = net.layers('conv5_stage1').params(2).get_data();

result = output{1}
figure(1);
imshow(img);
N = 160;
cx = result(1)*N;
cy = result(2)*N;
width = result(3)^2*N;
height = result(4)^2*N;
rectangle('Position',[cx-width*0.5,cy-height*0.5,width,height],'EdgeColor','green');
