close all;
patch = im2double(rgb2gray(imread('average_face_eye.jpg')));
figure;
imshow(patch);

filename = '044_n1';
img = im2double(imread(strcat(filename, '.pgm')));
figure;
imshow(img);

results = normxcorr2(patch, img);
figure;
imshow(results);
imwrite(results, strcat('cross_corr_', strcat(filename, '.png')));
% [val, idx] = max(results, [], 2);
% [max_val, max_idx] = max(val);

% [v, ind] = max(results);
% [v1, ind1] = max(max(results));
% row = ind(ind1)
% col = ind1

