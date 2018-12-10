%% Author: Shashwat Srivastava
% This method uses the 'lossyness' of the JPEG image
% to detect if it has been digitally modified
% A fancy way to say that it does ELA on a JPEG image
% Prereq: The image inputted must be a JPEG image
%%
function final = ELA(img, main, back)
 %% Set-Up:
 % Extracting the central image window and saving it 
 % at different compression levels
 % Saving the background image at a fixed compression
 % rate of 85%
 %% Naive ELA Algorithm
 % Naive ELA: Taking the average squared difference between the
 % image and the compressed image across color channels

 imwrite(main, 'center.jpg', 'Quality', 90);
 % we used multiple compression rates
 imwrite(back, 'Background_image.jpg', 'Quality', 85);
 imwrite(img, 'Original_image.jpg', 'Quality', 85);
 center_rows = size(img,1)./2 - 100;
 center_cols =  size(img, 2)./2 - 100;
 back = imread('Background_image.jpg');
 img = imread('Original_image.jpg');
 main = imread('center.jpg');
 back(center_rows:1:(center_rows+200), ... 
     center_cols:1:(center_cols+200),:) = main;
diff = img - back;
final = (diff .^2)/3;
imshow(final);
imwrite(final,'ELA_90.jpg');


 %% Problem
 % Works fine for smooth images
 % But gets 'confused' by highly textured images like that of grass
 % Confused --> shows pixels which haven't been modified at all at
 % edges or wherever noise is high
 % Specifically, because the image difference is computed across all 
 % spatial frequencies, a region with small amounts of high spatial 
 % frequency content (e.g., a mostly uniform sky) will have a lower difference
 % as compared to a highly textured region (e.g., grass). (Farid)
 %% Optimization
 % Average the difference across a window of b*b pixels [b = 16 here]
 % Normal the end result so that the values are in the interval [0,1]
 
soln = zeros(size(img));
b = 16;
bmx = [];
bmy = [];
bx = 0:1:15;
by = (0:1:15)';
n = zeros(0, 1);
for i = 1:16
    bmx = [bmx; bx];
    bmy = [bmy by];
end

for row = 1:size(img,1)
    for col = 1:size(img, 2)
      
        if(size(find((row+bmx) > size(img,1)), 1) ~=0 || size(find((col+bmy) > size(img, 2)), 1) ~= 0)
            break;
        end
        sum1 = img(row+bmx, col+bmy, :) - back(row+bmx, col+bmy, :);
        soln(row, col, :) = sum(sum(sum1))/b.^2;
        soln(row, col,:) = sum(soln(row, col, :), 3)./3;
    end
end
new_soln = soln - (min(min(min(soln))))/(max(max(max(soln))) - min(min(min(soln))));

imwrite(soln,'ELA_Optimized.jpg');
figure;imshow(soln);
figure; imshow(new_soln);
imwrite(new_soln, 'ELA_test_soln_image.jpg');
final = soln;
end


