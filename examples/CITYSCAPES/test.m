img = imread('/media/fu/Elements/tmp/gtFine/val/lindau/lindau_000000_000019_gtFine_color.png');
imgInstance = imread('/media/fu/Elements/tmp/gtFine/val/lindau/lindau_000000_000019_gtFine_instanceIds.png');
imgClass = imread('/media/fu/Elements/tmp/gtFine/val/lindau/lindau_000000_000019_gtFine_labelIds.png');
imgReshape = imresize(img,[300,300],'nearest');
figure(1);subplot(2,1,1);imshow(img);
% bw = imgInstance==26012;
% figure(1);subplot(2,1,2);imshow(bw);
% c = mean(imgClass(find(bw)))
figure(1);subplot(2,1,2);imshow(imgReshape);