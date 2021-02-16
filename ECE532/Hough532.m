% Hough Transform

% % Read the Image
% I = imread('grayobjects.png');
% imshow(I)
% 
% % Compute Edge Map
% E = edge(I,'Canny', 0.15);
% imshow(E)

E = (imread('edges.png')>0);

% Hough Transform
[H,theta,rho] = hough(E);
figure
imshow(imadjust(rescale(H)),[],...
       'XData',theta,...
       'YData',rho,...
       'InitialMagnification','fit'); % plot the H array as a heat map
xlabel('\theta (degrees)')
ylabel('\rho')
axis on
axis normal 
hold on
colormap(gca,hot)

% Find Peaks in Hough Array
P = houghpeaks(H,80,'threshold',ceil(0.3*max(H(:)))); % find peaks in H
sprintf('%d peaks detected', length(P))
disp('X & Y coordinates of the peaks:')
P(1:5,:)
x = theta(P(:,2));
y = rho(P(:,1));
plot(x,y,'s','color','black'); % superimpose tiny black squares at the peaks

% Find Line Segments Corresponding to Peaks
lines = houghlines(E,theta,rho,P,'FillGap',5,'MinLength',7); % line segments
sprintf('%d line segments',length(lines))
lines(1)
lines(2)

% Superimpose the Line Segments Onto the Original Image
figure, imshow(E), hold on
max_len = 0;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1),xy(:,2),'LineWidth',3,'Color','green');
end