addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

solver = caffe.Solver('blackpoint_vgg_solver.prototxt');
solver.step(1);

data = solver.net.blobs('data').get_data();
label = solver.net.blobs('label').get_data();

idx = 3;

img = data(:,:,:,idx)';
figure(1);imshow(uint8(img))

l = label(:,:,:,idx);


% img = imread('/media/fu/Elements/tmp/brainwash/brainwash_10_27_2014_images/00012000_640x480.png');
% caffeImg = toCaffeImg(img);

% tic
% output = net.forward();
% toc

% w1 = net.layers('conv4_stage1').params(1).get_data();
% w2 = net.layers('conv5_stage1').params(2).get_data();

% result = output{1}
% figure(1);
% imshow(img);
% N = 640;
% cx = result(1)*N;
% cy = result(2)*N;
% width = result(3)^2*N;
% height = result(4)^2*N;
% rectangle('Position',[cx-width*0.5,cy-height*0.5,width,height]);
