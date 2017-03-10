function [ im_data ] = toCaffeImg( im )
%TOCAFFEIMG Summary of this function goes here
%   Detailed explanation goes here
% Convert an image returned by Matlab’s imread to im_data in caffe’s data
% format: W x H x C with BGR channels
    if(size(im,3)==3)
        im_data = im(:, :, [3, 2, 1]); % permute channels from RGB to BGR
    else
        im_data = im;
    end
    im_data = permute(im_data, [2, 1, 3]); % flip width and height
    im_data = single(im_data); % convert from uint8 to single
%     im_data = imresize(im_data, [IMAGE_DIM IMAGE_DIM], 'bilinear'); % resize 
end

