% Final Project - 3HAA C. elegans analysis 
% Samuel Freitas
% ECE 532 


% read in the data 
% isolate worm
% 3D plot rgb and hsv
% isolate 3HAA 
% find head 
% find body shape 

data_dir = 'GLS130_15day_RED';
data_dir = dir(fullfile(data_dir,'*.tif'));

for i = 1:length(data_dir)
    imgs{i} = imread(fullfile(data_dir(i).folder,data_dir(i).name));
end

img = imgs{7};
img = imgs{14};

redChannel = img(:,:,1); % Red channel
greenChannel = img(:,:,2); % Green channel
blueChannel = img(:,:,3); % Blue channel


[countsR, binlocationsR] = imhist(redChannel, 255);
[countsG, binlocationsG] = imhist(greenChannel, 255);
[countsB, binlocationsB] = imhist(blueChannel, 255);
T = otsuthresh(countsR);

% imwrite(redChannel,'test.png')

bw = (redChannel<150);
bw_segmented = bwareafilt(bw,[100000 10000000000]);
bw_full = imfill(bw_segmented, 'holes');

% bw_skel = bwskel(bw_full,'MinBranchLength',1000);
% mask = imdilate(bw_skel,strel('disk',75));

mask = imopen(bw_full,strel('disk',10));

for i = 1:3
    img_mask(:,:,i) = (double(img(:,:,i)).*double(mask))/double(max(max(img(:,:,i))));
end

figure
imshow(img_mask)

redVals = nonzeros(img_mask(:,:,1));
greenVals = nonzeros(img_mask(:,:,2));
blueVals = nonzeros(img_mask(:,:,3));

img_hsv = rgb2hsv(img_mask);

h = nonzeros(img_hsv(:,:,1));
s = nonzeros(img_hsv(:,:,2));
v = nonzeros(img_hsv(:,:,3));

figure
hold on
for i = 1:500:length(redVals)
    colors = [redVals(i),greenVals(i),blueVals(i)];
    scatter(h(i),s(i),'filled','MarkerFaceColor',colors)
end
hold off

hsv_mask = (img_hsv(:,:,1)<0.2).*(img_hsv(:,:,2)>0.3);

for i = 1:3
    img_mask2(:,:,i) = (double(img(:,:,i)).*double(hsv_mask))/double(max(max(img(:,:,i))));
end

redVals2 = nonzeros(img_mask(:,:,1));
greenVals2 = nonzeros(img_mask(:,:,2));
blueVals2 = nonzeros(img_mask(:,:,3));

figure
hold on
for i = 1:500:length(redVals2)
    colors = [redVals2(i),greenVals2(i),blueVals2(i)];
    scatter3(redVals2(i),greenVals2(i),blueVals2(i),300,'filled','MarkerFaceColor',colors)
end
hold off

imshow(bw_full)

% [sortedAreas, sortIndexes] = sort([blobMeasurements.Area], 'descend');
% nonzeros(sortIndexes(sortedAreas>100000))

% imclose(imdilate(A>0,strel('disk',10)),strel('disk',50))

% 
% cd('python_dir')
% [a,b] = system('py hello_there.py');
% cd('..')