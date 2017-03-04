folder = '/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2012/';
table = readtable([folder 'Leaf_counts.csv'],'Delimiter',',','ReadVariableNames',false);

for i = 1:size(table,1)
    aFile = table{i,1};
    aFile = aFile{1};
    num = table{i,2};
    img = imread([folder aFile '_rgb.png']);
    label = imread([folder aFile '_label.png']);
    boxes = csvread([folder aFile '_bbox.csv']);
    imwrite(label,[folder aFile '_label.bmp']);
    
%     figure(1);
%     imshow(img);
%     hold on;
%     plot(boxes(:,2:2:end)',boxes(:,3:2:end)');
%     hold off;

    height = size(img,1);
    width = size(img,2);
    M = zeros(num,5);
    for n = 1:num
        aLabel = boxes(n,1);
        [ys,xs] = find(label==aLabel);
        x1 = min(xs);
        y1 = min(ys);
        x2 = max(xs);
        y2 = max(ys);
%         hold on;
%         rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','green');
%         hold off;
        M(n,:) = [aLabel, x1/width, y1/height, x2/width, y2/height];
    end
    
    csvwrite([folder aFile '_box.csv'],M);
end