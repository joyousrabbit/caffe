function [ img ] = fillRectangle( img,x1,y1,x2,y2,v )
%FILLCIRCLE Summary of this function goes here
%   Detailed explanation goes here
    img(y1:y2,x1:x2) = v;
end

