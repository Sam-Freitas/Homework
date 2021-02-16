% Homework # 4 - Kittlers method thresholding 
% Samuel Freitas
% ECE 532 

% read in the image(s)
img = imread('address.png');
img2 = imread('grayobjects.png');

% call the function without a user treshold
[kittlerThreshold,H,thresholded_img] = HW4_thresh(img);

% show what happens when there is no threshold provided
figure;
subplot(2,2,1); plot(H); 
title('When using kittlers method, H plot returned');
subplot(2,2,2); imshow(thresholded_img); 
title(['On threshold ' num2str(kittlerThreshold)])

imwrite(thresholded_img,'HW4_kittlers_thresh.png')

% call the function with a user threshold
[userThreshold,userH,thresholded_img] = HW4_thresh(img,100);

% show what happens when there is a threshold provided
subplot(2,2,3); plot(userH); 
title('When user defined threshold, image histogram plot returned');
subplot(2,2,4); imshow(thresholded_img); 
title(['On user threshold ' num2str(userThreshold)])
