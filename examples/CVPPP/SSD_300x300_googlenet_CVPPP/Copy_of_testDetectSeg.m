% export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
clear;
addpath('/home/fu/workspace/deeplearning/caffe-ssd/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

deployFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300_googlenet_CVPPP/deploy.prototxt';
modelFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/SSD_300x300_googlenet_CVPPP/VGG_VOC0712_SSD_300x300_iter_40000.caffemodel';
netDetect = caffe.Net(deployFile,modelFile,'test');

img = imread('/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/ara2013_plant155_rgb.png');
% img = permute(img,[2,1,3]);
% img = imrotate(img,180);

N = 300;
img = imresize(img,[N,N]);

% figure(1);
% imshow(uint8(img));

% subplot(2,1,2);
% imshow(img);

exMem = zeros(size(img,1),size(img,2));

caffeImgDetect = toCaffeImg(img);
caffeImgDetect(:,:,end+1) = exMem;

tic
output = netDetect.forward({caffeImgDetect});
toc

result = output{1};

firstTh = 0.9;

figure(1);
subplot(2,1,1);
imshow(img);
hold on;
for maxInd = 1:size(result,2)
    if result(3,maxInd)>firstTh
        xy = result(4:7,maxInd)*N;
        pw = xy(3)-xy(1);
        ph = xy(4)-xy(2);
        pcx = (xy(1)+xy(3))*0.5;
        pcy = (xy(2)+xy(4))*0.5;
        rectangle('Position',[xy(1),xy(2),pw,ph],'EdgeColor','red');
    end
end
hold off;

labels = result(4:7,result(3,:)>firstTh)';
labels = round(labels.*repmat([size(img,2),size(img,1),size(img,2),size(img,1)],size(labels,1),1));

deployFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/Deconv_48x48/deploy.prototxt';
modelFile = '/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/Deconv_48x48/VGG_VOC0712_SSD_48x48_iter_10000.caffemodel';
netSeg = caffe.Net(deployFile,modelFile,'test');

% ind = randi(size(labels,1));
for ind = 1:size(labels,1)
    for fineSegIt = 1:5
        labels = round(labels);
        
        figure(1);
        subplot(2,1,1);
        imshow(img);
        hold on;
        for maxInd = 1:size(labels,1)
            rectangle('Position',[labels(maxInd,1),labels(maxInd,2),labels(maxInd,3)-labels(maxInd,1)+1,labels(maxInd,4)-labels(maxInd,2)+1],'EdgeColor','red');
        end
        hold off;
        
        x1 = labels(ind,1);
        y1 = labels(ind,2);
        x2 = labels(ind,3);
        y2 = labels(ind,4);

        figure(1);subplot(2,1,1);
        rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','blue');

        segImg = img(y1:y2,x1:x2,:);
        segImg_reshape = imresize(segImg,[48,48]);
        caffeImgSeg = toCaffeImg(segImg_reshape);
        caffeImgSeg(:,:,end+1) = 0;

        tic
        output = netSeg.forward({caffeImgSeg});
        toc

        result = output{1}';
        result = imresize(result,[size(segImg,1),size(segImg,2)]);

        bw = result>200;

        figure(1);
        subplot(2,1,2);
    %     imshow(result/max(result(:)));
        imshow(bw);
        axis equal;
    %     pause(2);

        if fineSegIt<5
            [ys,xs] = find(bw);
            minX = min(xs);
            maxX = max(xs);
            minY = min(ys);
            maxY = max(ys);

            x1 = labels(ind,1)+minX-1;
            y1 = labels(ind,2)+minY-1;
            x2 = labels(ind,3)-(size(segImg,2)-maxX);
            y2 = labels(ind,4)-(size(segImg,1)-maxY);

            cx = (x1+x2)*0.5;
            cy = (y1+y2)*0.5;
            w = x2-x1;
            h = y2-y1;

            w = w*1.1;
            h = h*1.1;
            x1 = cx-0.5*w;
            y1 = cy-0.5*h;
            x2 = cx+0.5*w;
            y2 = cy+0.5*h;

            labels(ind,1) = max(x1,1);
            labels(ind,2) = max(y1,1);
            labels(ind,3) = min(x2,size(img,2));
            labels(ind,4) = min(y2,size(img,1));
        end
    end
    
    exMem(y1:y2,x1:x2,:) = max(bw*ind,exMem(y1:y2,x1:x2,:));
end

imagesc(exMem);
axis equal;


mem = exMem>0;
% se = strel('square',5);
% mem = imdilate(mem,se);
mem = mem*255;
caffeImgDetect(:,:,end) = mem';
tic
output = netDetect.forward({caffeImgDetect});
toc

result = output{1};

secondTh = 0.4;

figure(1);
subplot(2,1,1);
imshow(img);
hold on;
for maxInd = 1:size(result,2)
    if result(3,maxInd)>secondTh
        xy = result(4:7,maxInd)*N;
        pw = xy(3)-xy(1);
        ph = xy(4)-xy(2);
        pcx = (xy(1)+xy(3))*0.5;
        pcy = (xy(2)+xy(4))*0.5;
        rectangle('Position',[xy(1),xy(2),pw,ph],'EdgeColor','red');
    end
end
hold off;

labels = result(4:7,result(3,:)>secondTh)';
labels = round(labels.*repmat([size(img,2),size(img,1),size(img,2),size(img,1)],size(labels,1),1));

for ind = 1:size(labels,1)
    for fineSegIt = 1:5
        labels = round(labels);
        
        figure(1);
        subplot(2,1,1);
        imshow(img);
        hold on;
        for maxInd = 1:size(labels,1)
            rectangle('Position',[labels(maxInd,1),labels(maxInd,2),labels(maxInd,3)-labels(maxInd,1)+1,labels(maxInd,4)-labels(maxInd,2)+1],'EdgeColor','red');
        end
        hold off;
        
        x1 = labels(ind,1);
        y1 = labels(ind,2);
        x2 = labels(ind,3);
        y2 = labels(ind,4);

        figure(1);subplot(2,1,1);
        rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','blue');

        segImg = img(y1:y2,x1:x2,:);
        segImg_reshape = imresize(segImg,[48,48]);
        caffeImgSeg = toCaffeImg(segImg_reshape);
        caffeImgSeg(:,:,end+1) = 0;

        tic
        output = netSeg.forward({caffeImgSeg});
        toc

        result = output{1}';
        result = imresize(result,[size(segImg,1),size(segImg,2)]);

        bw = result>200;

        figure(1);
        subplot(2,1,2);
    %     imshow(result/max(result(:)));
        imshow(bw);
        axis equal;
    %     pause(2);

        if fineSegIt<5
            [ys,xs] = find(bw);
            minX = min(xs);
            maxX = max(xs);
            minY = min(ys);
            maxY = max(ys);

            x1 = labels(ind,1)+minX-1;
            y1 = labels(ind,2)+minY-1;
            x2 = labels(ind,3)-(size(segImg,2)-maxX);
            y2 = labels(ind,4)-(size(segImg,1)-maxY);

            cx = (x1+x2)*0.5;
            cy = (y1+y2)*0.5;
            w = x2-x1;
            h = y2-y1;

            w = w*1.1;
            h = h*1.1;
            x1 = cx-0.5*w;
            y1 = cy-0.5*h;
            x2 = cx+0.5*w;
            y2 = cy+0.5*h;

            labels(ind,1) = max(x1,1);
            labels(ind,2) = max(y1,1);
            labels(ind,3) = min(x2,size(img,2));
            labels(ind,4) = min(y2,size(img,1));
        end
    end
    
    exMem(y1:y2,x1:x2,:) = max(bw*ind,exMem(y1:y2,x1:x2,:));
end

imagesc(exMem);
axis equal;