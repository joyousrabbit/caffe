addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = 'blackpoint_lstm_deploy.prototxt';
% modelFile = '/media/fu/Elements/nul/blackpoint_lstm_3t_iter_100000.caffemodel';
modelFile = '/media/fu/Elements/nul/blackpoint_lstm_iter_118742.caffemodel';
net = caffe.Net(deployFile,modelFile,'test');

img = zeros(160,160)+0;
% img = fillCircle(img,100,50,20,210);
% img = fillCircle(img,50,100,30,100);
% 
% img = fillCircle(img,100,50,30,210);
% img = fillCircle(img,100,80,30,100);

img = fillCircle(img,100,30,20,210);
img = fillCircle(img,80,100,40,100);

% img = fillRectangle(img,50,50,60,60,200);
caffeImg = toCaffeImg(img);

tic
output = net.forward({caffeImg});
toc

% w1 = net.layers('conv4_stage1').params(1).get_data();
% w2 = net.layers('conv5_stage1').params(2).get_data();

results = output{1}
figure(1);
imshow(uint8(img));
N = 160;
hold on;
for i = 1:size(results,1)
    result = results(i,:);
    if result(end)>0.5
        width = exp(result(3)*log(N));
        height = exp(result(4)*log(N));
        cx = result(1)*N;
        cy = result(2)*N;
        rectangle('Position',[cx-width*0.5,cy-height*0.5,width,height],'EdgeColor','green');
    end
end
hold off;
