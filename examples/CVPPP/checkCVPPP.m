folder = '/media/fu/Elements/tmp/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2013-Canon/';
table = readtable([folder 'Leaf_counts.csv'],'Delimiter',',','ReadVariableNames',false);

for i = 1:size(table,1)
    aFile = table{i,1}
    aFile = aFile{1};
    num = table{i,2};
    img = imread([folder aFile '_rgb.png']);
    boxes = csvread([folder aFile '_box.csv']);
    
    height = size(img,1);
    width = size(img,2);
    
    size(img)
    
    figure(1);
    subplot(1,1,1);
    imshow(img);
    hold on;
    for n = 1:size(boxes,1)
        x1 = boxes(n,2)*width;
        y1 = boxes(n,3)*height;
        x2 = boxes(n,4)*width;
        y2 = boxes(n,5)*height;
        rectangle('Position',[x1,y1,x2-x1,y2-y1],'EdgeColor','red');
    end
    hold off;
    pause(5)
end