addpath('/home/fu/workspace/deeplearning/caffe-ssd/models/VGGNet/VOC0712/CITYSCAPE/jsonlab-1.5');

rootPath = '/home/fu/workspace/dataset/leftImg8bit/train/';

validLabels = {'person','rider','car','truck','bus','train','motorcycle','bicycle'};
validLabelsIDs = [24,25,26,27,28,31,32,33];

% fid = fopen([rootPath 'list.txt'],'wt');

maxObjects = 0;

allFiles = getAllFiles(rootPath);
for i = 1:length(allFiles)
    aFile = allFiles{i};
    if aFile(end-4:end) == '.json' 
        filePre = aFile(1:end-21)
%         fprintf(fid,'%s\n',filePre);

        M = [];
        img = imread([filePre '_leftImg8bit.png']);
        img = imresize(img,0.5);
        img = img(:,1:end/2,:);
        
        height = size(img,1);
        width = size(img,2);
%         labels = imread([filePre '_gtFine_labelIds.png']);
        instances = imread([filePre '_gtFine_instanceIds.png']);
        instances = imresize(instances,0.5,'nearest');
        instances = instances(:,1:end/2,:);
        
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
                x1 = min(xs);
                y1 = min(ys);
                x2 = max(xs);
                y2 = max(ys);
                if length(ys)>500 && x1+2<x2 && y1+2<y2 % should have some surface
%                 figure(1);subplot(1,1,1);imshow(img);
%                 hold on;
%                 rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','green');
%                 hold off;
                
                    M(end+1,:) = [anInstance,ind,x1/width,y1/height,x2/width,y2/height];
                end
            end
        end
        
%         js = loadjson(aFile);
%         for j = 1:length(js.objects)
%             o = js.objects{j};
%             ind = strmatch(o.label,validLabels,'exact');
%             if ind>0
%                 x1 = min(o.polygon(:,1));
%                 y1 = min(o.polygon(:,2));
%                 x2 = max(o.polygon(:,1));
%                 y2 = max(o.polygon(:,2));
%                 figure(1);subplot(1,1,1);imshow(img);
%                 hold on;
%                 rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','green');
%                 hold off;
%             end
%         end

        imwrite(img,[filePre '_leftImg8bit.left.png']);
        imwrite(instances,[filePre '_gtFine_instanceIds.left.png']);
        
        csvwrite([filePre '_gtFine_box.left.csv'],M);
        maxObjects = max(maxObjects,size(M,1));
    end
end

% fclose(fid);

maxObjects