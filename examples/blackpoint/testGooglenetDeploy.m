addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = 'blackpoint_googlenet_deploy.prototxt';
modelFile = '/media/fu/Elements/nul/blackpoint_googlenet_iter_90345.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

img = zeros(160,160)+0;
img = fillCircle(img,100,50,20,210);
img = fillCircle(img,40,80,20,200);
% img = fillRectangle(img,50,50,60,60,200);
caffeImg = toCaffeImg(img);

tic
output = net.forward({caffeImg});
toc

% w1 = net.layers('conv4_stage1').params(1).get_data();
% w2 = net.layers('conv5_stage1').params(2).get_data();

result = output{1}
figure(1);
imshow(uint8(img));
N = 160;
width = exp(result(3)*log(N));
height = exp(result(4)*log(N));
cx = result(1)*N;
cy = result(2)*N;
rectangle('Position',[cx-width*0.5,cy-height*0.5,width,height],'EdgeColor','green');
