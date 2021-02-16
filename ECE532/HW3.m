% Homework # 3 - Hough transform 
% Samuel Freitas
% ECE 532 

% FINAL deliveries (post running)
% H - Hough array as a grayscale 100x100 image from 0:255
% num_significant_lines - number of significant lines detected
% H_threshold - threshold value used 
% lines - line value parameters stored as a struct with rho/theta/coords
% displayed image - top: hough array with 'hot' colormap
%                 - bottom: significant lines overlayed on inital image

clear all; close all
% edges.png must be in the same folder
I = imread('edges.png');

% flip image vertically to align orgin
I2 = flipud(I);

% find all the nonzero values in the image
[x,y,vals] = find(I2);

% use the long side of the image to find N
N = length(I);

% from lecture 9/22
% theta = 0:pi
% rho = -N:N*sqrt(2)
theta = linspace(0,pi,100);
rho = linspace(-N,N*sqrt(2),100);

% initalize the hough matrix
H = zeros(length(rho),length(theta));

% iterate through all the values
for i = 1:length(vals)
    
    % iterate through theta
    for j = 1:length(theta)
        
        % calculate a rho value for theta(j) and coord(i)
        rhoThis = x(i)*cos(theta(j)) + y(i)*sin(theta(j));
        
        % find the closest representative from the inital rho values
        [~,rhoIdx] = min(abs(rho-rhoThis));
        
        % iterate that rho/theta coordinate 
        H(rhoIdx,j) = H(rhoIdx,j) +1;
        
    end
    
end

% using the top 65% of the max value as the threshold
H_threshold = ceil(0.35*max(H(:)));

% take only the top 50 peaks from the top 65% threshold
P = houghpeaks(H,50,'threshold',H_threshold); 
sprintf(['%d peaks detected over the threshold of ' num2str(H_threshold)], length(P))

% use a fillgap of 2 and a minlength of 5 to find only quality lines
% the amount is enough to generally reconstruct the center image ~50 lines
lines = houghlines(I,theta,rho,P,'FillGap',2,'MinLength',5);
num_significant_lines = length(lines);
sprintf([ num2str(num_significant_lines) ' lines detected from those peaks'] )

% scale H to 0:255
H = 255*(H/max(H(:)));
% scale for saving and writing
imwrite(H/max(H(:)),'HW3_hough_array.png');

% display the hough transform
img = figure('units','normalized','outerposition',[0 0 1 1]);
subplot(2,1,1)
imshow(H/max(H(:))); colormap hot
xticks_numbers = (1:10:100);
yticks_numbers = (1:10:100);
xticks_labels = string(theta(1:10:end));
yticks_labels = string(rho(1:10:end));
axis on
set(gca,'XTick',xticks_numbers)
set(gca,'XTickLabels',xticks_labels)
set(gca,'YTick',yticks_numbers)
set(gca,'YTickLabels',yticks_labels)
xlabel('\theta'), ylabel('\rho');
title('Hough Transform of the Edge Image')

% display the line sketch 
subplot(2,1,2)
imshow(I); title(['image of ' num2str(num_significant_lines) ' most significant lines detected in image "edges.png"'])
hold on
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',3,'Color','yellow');
end

hold off
axis off

pause(.2)

imwrite(getframe(gcf).cdata,'HW3_Hough_array_and_lines.png')
save('HW3_output_variable.mat','H','H_threshold','num_significant_lines','lines');

clear rhoIdx rhoThis i j xy max_len k 


