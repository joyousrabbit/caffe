addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

solver = caffe.Solver('ssd_googlenet_solver.prototxt');
solver.step(1);

data = solver.net.blobs('data').get_data();
label = solver.net.blobs('label').get_data();
priorBoxes = solver.net.blobs('mbox_priorbox').get_data();

idx = 1;
pos = 67;

img = data(:,:,:,idx)';
figure(1);imshow(uint8(img))
% hold on;
% pb = priorBoxes(:,pos,1);
% x1 = pb(1)*size(img,2);
% y1 = pb(2)*size(img,1);
% x2 = pb(3)*size(img,2);
% y2 = pb(4)*size(img,1);
% rectangle('Position',[x1,y1,x2-x1,y2-y1],'edgeColor','green');
% hold off;

l = label(:,:,:,idx);

% 
% Num: 0
% matched: 45->0
% matched: 46->0
% matched: 47->0
% Num: 1
% matched: 171->0
% matched: 172->0
% Num: 2
% matched: 54->1
% matched: 56->1
% matched: 84->1
% matched: 86->1
% matched: 336->0
% matched: 337->0
% matched: 338->0
% Num: 3
% matched: 258->0
% matched: 333->1
% matched: 334->1
% Num: 4
% matched: 141->0
% matched: 142->0
% matched: 143->0
% matched: 321->1
% matched: 322->1
% matched: 323->1
% matched: 378->1
% matched: 380->1
% Num: 5
% matched: 234->1
% matched: 235->1
% matched: 236->1
% matched: 348->0
% matched: 349->0
% matched: 352->0
% Num: 6
% matched: 252->1
% matched: 253->1
% matched: 254->1
% matched: 354->0
% Num: 7
% matched: 336->0
% matched: 387->0
% matched: 388->0
% matched: 389->0
% matched: 402->1