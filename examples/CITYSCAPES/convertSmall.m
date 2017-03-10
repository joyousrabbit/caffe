addpath('/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/CITYSCAPE/jsonlab-1.5');

rootPath = '/home/fu/workspace/dataset/leftImg8bit/train/';

allFiles = getAllFiles(rootPath);
for i = 1:length(allFiles)
    aFile = allFiles{i};
    if aFile(end-4:end) == '.json' 
        filePre = aFile(1:end-21);
        
        img = imread([filePre '_leftImg8bit.png']);
        height = size(img,1);
        width = size(img,2);
        labels = imread([filePre '_gtFine_labelIds.png']);
        instances = imread([filePre '_gtFine_instanceIds.png']);
        
        imgSmall = imresize(img,0.5);
        instancesSmall = imresize(instances,0.5,'nearest');
        
        imwrite(imgSmall,[filePre '_leftImg8bit.small.png']);
        imwrite(instancesSmall,[filePre '_gtFine_instanceIds.small.png']);
    end
end
