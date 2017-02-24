function [ img ] = fillCircle( img,cx,cy,r,v )
%FILLCIRCLE Summary of this function goes here
%   Detailed explanation goes here
    for i = 1:size(img,1)
        for j = 1:size(img,2)
            if(norm([i,j]-[cy,cx])<=r)
                img(i,j) = v;
            end
        end
    end
end

