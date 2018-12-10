%% Author: Shashwat Srivastava
% ssrivastav26@wisc.edu
% This is an implementation of the ELA
% Heavily inspired and based on the methods 
% proposed Hany Farid
%%

%%
% read the image and calculate the
% location of the center pixels
% extract a central window of 200 x 200 pixels
% color the pixels in the background black
% send it to ELA.m method for Error Level Analysis
img = imread('textured.jpg');
center_rows = size(img,1)./2 - 100;
center_cols =  size(img, 2)./2 - 100;
center = img(center_rows:1:(center_rows+200), ... 
    center_cols:1:(center_cols+200),:); 
back = img;
back(center_rows:1:(center_rows+200), ... 
    center_cols:1:(center_cols+200),:) = 0;
imshow(center);   
final = ELA(img, center, back);
%%