filePre = '/home/fu/workspace/dataset/leftImg8bit/train/cologne/cologne_000009_000019';

M = [];
img = imread([filePre '_leftImg8bit.small.png']);
height = size(img,1);
width = size(img,2);
% labels = imread([filePre '_gtFine_labelIds.png']);
instances = imread([filePre '_gtFine_instanceIds.small.png']);
uniqueInstances = double(unique(instances(:)));
for j = 1:length(uniqueInstances)
    anInstance = uniqueInstances(j);
    classID = anInstance;
    if classID>1000
        classID = floor(classID/1000);
    end
    ind = find(validLabelsIDs==classID);
    if ~isempty(ind)
        bw = instances==anInstance;
        [ys,xs] = find(bw);
        length(ys)
        x1 = min(xs);
        y1 = min(ys);
        x2 = max(xs);
        y2 = max(ys);
        figure(1);
        subplot(2,1,1);imshow(img);
        hold on;
        rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','green');
        hold off;
        subplot(2,1,2);imshow(bw);
        
        anInstance
%         ind
%         [x1/width,y1/height,x2/width,y2/height]
    end
end