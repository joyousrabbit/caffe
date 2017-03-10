% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
clear;
addpath('/home/fu/workspace/deeplearning/caffe-ssd/matlab');

% colors = rand(100,3);
% meanColors = mean(colors,2);
% colors(meanColors<0.5,:) = [];
% stdColors = std(colors,0,2);
% colors(stdColors<0.15,:)=[];
% save('colors','colors');
load('colors');
colors = [colors;colors];


caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

detectDeployFile = 'deploy.prototxt';
detectModelFile = 'CITYSCAPE_SSD_300x300_left_iter_120000.caffemodel';
detectNet = caffe.Net(detectDeployFile,detectModelFile,'test');

segDeployFile = 'deploySeg.prototxt';
segModelFile = 'seg_48x48_iter_120000.caffemodel';
segNet = caffe.Net(segDeployFile,segModelFile,'test');


N = 300;
maxDetectIter = 4;
detThs = [0.9, 0.7, 0.6, 0.5];
segThs = [0.6, 0.5, 0.5, 0.5];
% maxDetectIter = 1;
% detThs = [0.9];
% segThs = [0.5];
maxSegIter = 10;

% testList = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300_googlenet_CVPPP/Leaf_counts_testing.csv';
testList = '/home/fu/workspace/dataset/leftImg8bit/val/list.txt';
fid = fopen(testList);
tline = fgetl(fid);
while ischar(tline)
% Begin of test loop

% img = imread('/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2012/ara2012_plant118_rgb.png');
% oriImg = imread('/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/A1_test/plant160_rgb.png');
oriImg = imread([tline '_leftImg8bit.png']);
oriImg = oriImg(:,1:end/2,:);
img = imresize(oriImg,[N,N]);
exMem = zeros(size(img,1),size(img,2));
instancesMem = zeros(size(img,1),size(img,2));

caffeImg = toCaffeImg(img);
caffeImgWithMem = cat(3,caffeImg,exMem');

instanceInd = 0;
for detectIter = 1:maxDetectIter
    tic
    outputDetections = detectNet.forward({caffeImgWithMem});
    toc
    
    outputDetections = outputDetections{1};
    detections = outputDetections(4:7,outputDetections(3,:)>detThs(detectIter))';
    detections = round(detections*N);
    detections(:,1:2) = max(detections(:,1:2)-10,1);
    detections(:,3:4) = min(detections(:,3:4)+10,N);
    ws = detections(:,3)-detections(:,1);
    hs = detections(:,4)-detections(:,2);
    detections(ws<=0|hs<=0,:) = [];
    
    figure(1);
    subplot(2,1,1);
    imshow(img);
    hold on;
    for ind = 1:size(detections,1)
        aDetection = detections(ind,:);
        rectangle('Position',[aDetection(1),aDetection(2),aDetection(3)-aDetection(1)+1,aDetection(4)-aDetection(2)+1],'EdgeColor','red');
    end
    hold off;
    
    itMem = zeros(size(img,1),size(img,2));
    
    for ind = 1:size(detections,1)
        goodSeg = true;
        for fineSegIt = 1:maxSegIter
%             figure(1);
%             subplot(2,1,1);
%             imshow(img);
%             hold on;
%             for it = 1:size(detections,1)
%                 aDetection = detections(it,:);
%                 rectangle('Position',[aDetection(1),aDetection(2),aDetection(3)-aDetection(1)+1,aDetection(4)-aDetection(2)+1],'EdgeColor','red');
%             end
%             hold off;

            aDetection = detections(ind,:);
            x1 = aDetection(1);
            y1 = aDetection(2);
            x2 = aDetection(3);
            y2 = aDetection(4);
            w = x2-x1;
            h = y2-y1;
            if w<=0 || h<=0
                goodSeg = false;
                break;
            end

            figure(1);
            subplot(2,1,1);
            rectangle('Position',[x1,y1,w,h],'EdgeColor','blue');

            segCaffeImgWihtMem = caffeImgWithMem(x1:x2,y1:y2,:);
            segCaffeImgWihtMem_scale = imresize(segCaffeImgWihtMem,[48,48]);

            tic
            outputSeg = segNet.forward({segCaffeImgWihtMem_scale});
            toc

            outputSeg = outputSeg{1}';
            segScaledBack = imresize(outputSeg,[size(segCaffeImgWihtMem,2),size(segCaffeImgWihtMem,1)]);

%             bw = segScaledBack>max(max(segScaledBack(:))*0.7,255*0.5);
            bw = segScaledBack>255*segThs(detectIter);

            figure(1);
            subplot(2,1,2);
            imshow(bw);
            imshow(segScaledBack/max(segScaledBack(:)));

            [ys,xs] = find(bw);
            if isempty(ys) || sum(bw(:))<=36
                goodSeg = false;
                break;
            end
            minX = min(xs);
            maxX = max(xs);
            minY = min(ys);
            maxY = max(ys);

            x1new = x1+minX-1;
            y1new = y1+minY-1;
            x2new = x2-(size(bw,2)-maxX);
            y2new = y2-(size(bw,1)-maxY);

            cx = (x1new+x2new)*0.5;
            cy = (y1new+y2new)*0.5;
            w = x2new-x1new;
            h = y2new-y1new;
            
            if w<=0 || h<=0
                goodSeg = false;
                break;
            end

            w = w*1.1;
            h = h*1.1;
            x1new = max(round(cx-0.5*w),1);
            y1new = max(round(cy-0.5*h),1);
            x2new = min(round(cx+0.5*w),size(img,2));
            y2new = min(round(cy+0.5*h),size(img,1));
            
            if all(detections(ind,:)==[x1new,y1new,x2new,y2new])
                break;
            end

            detections(ind,1) = x1new;
            detections(ind,2) = y1new;
            detections(ind,3) = x2new;
            detections(ind,4) = y2new;
        end
        if goodSeg
            instanceInd = instanceInd+1;
            instancesMem(y1:y2,x1:x2) = max(bw*instanceInd,instancesMem(y1:y2,x1:x2));
            se = strel('square',3);
            bw = imdilate(bw,se);
            itMem(y1:y2,x1:x2) = max(bw*255,itMem(y1:y2,x1:x2));
        end
    end
    caffeImgWithMem(:,:,end) = max(itMem',caffeImgWithMem(:,:,end));
    subplot(2,1,2);
    imshow(caffeImgWithMem(:,:,end)');
end

figure(1);
subplot(2,1,1);
imshow(img);

coloredInstanceImg = zeros(size(instancesMem,1),size(instancesMem,2),3);
for i = 1:size(instancesMem,1)
    for j = 1:size(instancesMem,2)
        if instancesMem(i,j)>0
            coloredInstanceImg(i,j,:) = colors(instancesMem(i,j),:);
        end
    end
end
figure(1);subplot(2,1,2);
imshow(coloredInstanceImg);

% imwrite(uint8(imresize(instancesMem,[size(oriImg,1),size(oriImg,2)],'nearest')),[tline '_label.png']);

% End of test loop
    disp(tline)
    tline = fgetl(fid);
    pause(0.1);
 end
fclose(fid);
