addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = 'lstm_ex_bits_deploy.prototxt';
modelFile = '/media/fu/Elements/nul/lstm_ex_bits_iter_300000.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

num1 = 93;
num2 = 43;
s = num1+num2;

bitsNum1 = de2bi(num1,8);
bitsNum2 = de2bi(num2,8);
bitsS = de2bi(s,8);

data = single(zeros(1,2,1,8));
data(:,1,:,:) = bitsNum1(:);
data(:,2,:,:) = bitsNum2(:);

tic
output = net.forward({data});
toc

result = output{1}
bits = result(:)'>0.5;
