% Homework # 2 - Rodriguez edge detector 
% Samuel Freitas
% ECE 532 

% given variables
gradient_thresh = 60;

% these are given with starting index = 0
% matlab's index starts at 1
% region 1 -- 205 ≤ r ≤ 209, 182 ≤ c ≤ 186
% region 2 -- 347 ≤ r ≤ 351, 350 ≤ c ≤ 354

% read the horse image in and covert to a double for ease of use
% this is assuming the 'horse.png' is in the same folder
I = double(imread('horse.png'));

[rows_img,cols_img] = size(I);

% create Iy and Ix
Iy = zeros(size(I));
Ix = zeros(size(I));

% skip the outside edges and leave them as zeros
for r = 2:(rows_img-1)
   % iterate through the rows
   for c = 2:(cols_img-1)
       % iterate through the colums
       
       % compute Iy
       Iy_region_1 = median([I(r,c-1),I(r,c+1),I(r-1,c-1),I(r-1,c),I(r-1,c+1)]);
       Iy_region_2 = median([I(r,c-1),I(r,c+1),I(r+1,c-1),I(r+1,c),I(r+1,c+1)]);
       
       Iy(r,c) = Iy_region_1-Iy_region_2;
       
       % compute Ix
       Ix_region_1 = median([I(r-1,c),I(r-1,c+1),I(r,c+1),I(r+1,c),I(r+1,c+1)]);
       Ix_region_2 = median([I(r-1,c-1),I(r-1,c),I(r,c-1),I(r+1,c-1),I(r+1,c)]);
       
       Ix(r,c) = Ix_region_1-Ix_region_2;
       
   end
end
% remove uncessary variables 
clear Iy_region_1 Iy_region_2 Ix_region_1 Ix_region_2 r c 

% compute G
G = (( (Ix.^2) + (Iy.^2) ).^(1/2) )/ 2;

% compute E
E = (G>=gradient_thresh);

% compute sobel gradient map and edge map
sobel_grad_map = imgradient(I,'sobel')/8;
sobel_edge_map = sobel_grad_map >= gradient_thresh;

% region 1 -- 349 =< r =< 353, 349 =< c =< 353

JJR_grad_mag_region = G(350:354, 350:354);
JJR_edge_map_region = E(350:354, 350:354);
sobel_grad_map_region = sobel_grad_map(350:354, 350:354);
sobel_edge_map_region = sobel_edge_map(350:354, 350:354);
imwrite(E,'HW2_JJR_edge.png');

figure; 
subplot(2,2,1); imshow(I,[]); title('inital horse')
subplot(2,2,2); imshowpair(sobel_edge_map,E); title('sobel (green) JRR (magenta) edge map differences')
subplot(2,2,3); imshow(E,[]); title('JRR horse Edge map');
subplot(2,2,4); imshow(sobel_edge_map,[]); title('sobel edge map')


disp('JRR gradient magnitude values')
disp(JJR_grad_mag_region)
disp('JRR edge map values')
disp(JJR_edge_map_region)
disp('Sobel gradient magnitude values')
disp(sobel_grad_map_region)
disp('Sobel edge map')
disp(sobel_edge_map_region)









