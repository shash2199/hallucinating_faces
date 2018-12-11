%% Author: Shashwat Srivastava
% reading the image

img = imread('books-orig.jpg');
img = imgaussfilt(img);
% initializing variables
N = size(img, 1) .* size(img, 2) .* size(img,3);
B = 64;
b = sqrt(B);
e = 0.01;
Q = 256;
Nn = 100;
Nf = 128;
Nd = 16;
Nb = (sqrt(N) - sqrt(B) + 1)^2;
Nb = int64(Nb);
rows = size(img, 1) + (B - mod(size(img,1),B));
col = size(img, 2) + (B - mod(size(img,2),B));
new_img = imresize(img, [rows, col]);
red = im2double(new_img(:, :, 1));
blue = im2double(new_img(:, :, 2));
green = im2double(new_img(:, :, 3));
% de-meaning all the values
if mean(red) ~= 0
    red = red - mean(red);
end
if mean(blue) ~= 0
    blue = blue - mean(blue);
end
if mean(green) ~= 0
    green = green - mean(green);
end
% calculating the eigenvalues and eigenvectors
% and also calculating the PCA
[coeff_r, ~, eigen_red] = pca(red);
[coeff_b, ~, eigen_blue] = pca(blue);
[coeff_g, ~, eigen_green] = pca(green);
eigenvalues = [eigen_red; eigen_blue; eigen_green];
sum_b = 0;
for i = 1:b
   sum_b = sum_b + eigenvalues(i, 1);
end
one_minus_e = 1 - e;
sum = 0;
Nt = 1;
% 
for j = 1:size(eigenvalues, 1)
    sum = sum + eigenvalues(j, 1);
    frac = sum/sum_b;
    frac = round(frac*1e2)/1e2;
      if(frac == one_minus_e)
          Nt = j;
          break;
      end
 end

 matrix_Ai = zeros(Nb, Nt,3);
 reference_matrix = zeros(Nb, Nt+2,3);
rows = rows./b;
col = col./b;
matrix_Ai = zeros(Nb, B,3);
reference_matrix = zeros(Nb, B+2,3);
c = 1;
for i = 1:b:(rows(1,1)-b)
    for j = 1:b:(col(1,1)-b)
        [coeff_r,score_r,latent_r,~,explained_r] = pca(red(i:i+b-1,j:j+b-1), 'Economy',false);
        [coeff_b,score_b,latent_b,~,explained_b] = pca(blue(i:i+b-1,j:j+b-1), 'Economy',false);
        [coeff_g,score_g,latent_g,~,explained_g] = pca(green(i:i+b-1,j:j+b-1), 'Economy',false);
        matrix_Ai(c, :, 1) = reshape(red(i:i+b-1, j:j+b-1)*coeff_r, [1, B]);
        reference_matrix(c, :, 1) = [matrix_Ai(c, :, 1) j i];
        matrix_Ai(c, :, 3) = reshape(blue(i:i+b-1, j:j+b-1)*coeff_b(:, :), [1, B]);
        reference_matrix(c, :, 3) = [matrix_Ai(c, :, 3) j i];
        matrix_Ai(c, :, 2) = reshape(green(i:i+b-1, j:j+b-1)*coeff_g(:, :), [1, B]);
        reference_matrix(c, :, 2) = [matrix_Ai(c, :, 2) j i];
        c = c + 1;
    end
end
matrix_Ai = floor(matrix_Ai./Q);
reference_matrix(:, 1:B, :) = floor(reference_matrix(:, 1:B, :)./Q);
S = zeros(size(reference_matrix));
S(:, :, 1) = sortrows(reference_matrix(:, :, 1), 1:B);
S(:, :, 2) = sortrows(reference_matrix(:, :, 2), 1:B);
S(:, :, 3) = sortrows(reference_matrix(:, :, 3), 1:B);
final_list = [];
for k = 1:3
    for i = 1:Nb
        for j = 1:Nb
           if(i == j)
               continue;
           end
           if(abs(i - j) <= Nn)
               a_i = reference_matrix(i,B+1,k);
               a_j = reference_matrix(i,B+2,k);
               b_i = reference_matrix(j,B+1,k);
               b_j = reference_matrix(j,B+2,k);
               dist = sqrt((a_i - b_i).^2 + (a_j - b_j).^2);
               if(dist < Nd)
                   continue;
               end
               if((a_i - b_i) > 0)
                   final = [a_i - b_i, a_j - b_j, a_i, a_j, b_i, b_j];
               end
               if((a_i - b_i) < 0)
                   final = [-a_i + b_i, a_j - b_j, a_i, a_j, b_i, b_j];
               end
               if(a_i == b_i)
                  final = [0, abs(a_j - b_j), a_i, a_j, b_i, b_j]; 
               end
               final_list = [final_list; final];
           end
           if(abs(i - j) > Nn)
           break;
           end
        end
    end
end
final_list = sortrows(final_list, 1:2);
Last_list = [];
num = 0;
check = 0;
for i = 1:size(final_list,1)
    for j = i:size(final_list, 1)
        if(final_list(i, 1:2) == final_list(j, 1:2))
            num = num + 1;
            check = 1;
        end
        if(final_list(i, 1:2) ~= final_list(j, 1:2))
            if (check == 1)
                check = 0;
                break;
            end
        end
            
    end
    if(num >= Nd)
        Last_list = [Last_list; final_list(i, :)];
    end
num = 0;    
end
final_img = zeros(size(img,1), size(img,2));

for i = 1:size(Last_list, 1)
   final_img(Last_list(i,3), Last_list(i,4)) = 255;
   final_img(Last_list(i,5), Last_list(i,6)) = 255;
end
figure; imshow(final_img);



