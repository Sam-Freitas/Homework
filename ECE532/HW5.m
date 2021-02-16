% Homework # 5 - connected component labeling 
% Samuel Freitas
% ECE 532 


% read in the images as binary arrays
keys = imbinarize(imread('keys.png'));
book = imbinarize(imread('book.png'));
USAF = imbinarize(imread('USAF-1951.jpg'));

% send them to the function
book_labels = HW5_bwLabel(book);
keys_labels = HW5_bwLabel(keys);
USAF_labels = HW5_bwLabel(USAF);

figure;
subplot(1,3,1)
imshow(book_labels,[])
title('Books BW image labeled')
subplot(1,3,2)
imshow(keys_labels,[])
title('Keys BW image labeled')
subplot(1,3,3)
imshow(USAF_labels,[])
title('USAF BW image labeled')

% these will be hard to see if viewed normally in the system viewer
% but will show up when asigning varaibles to them
imwrite(uint8(book_labels),'HW5_book_labels.png')
imwrite(uint8(keys_labels),'HW5_keys_labels.png')



function [gray_labels] = HW5_bwLabel(img)

% find the background value
% I am defining it as the top left pixel
img_backval = img(1,1);
% pad the image with an outline to not break 
img2 = padarray(img,[1 1],img_backval);

% if the background is 1 then use the complement to keep notation tidy
if img_backval == 1
    img2 = imcomplement(img2);
end

[rows,cols]=size(img2);

% set up iterative elements 
labels = zeros(size(img2));
nextlabel = 1;

for i = 2:(rows-1)
    for j = 2:(cols-1)
        
        % if not background
        if img2(i,j)
            
            % get the 'window' from the label matrix
            labelWindow = labels(i-1:i+1, j-1:j+1);
            
            % check if that pixel is already labeled
            if sum(labelWindow(:)) 
                
                % if it is, then use the min of the label window
                possibleLabels = nonzeros(labelWindow);
                labels(i,j) = min(possibleLabels);
                
                % store the connected labels for future processing
                for k = 1:length(possibleLabels)
                   thisLabel = possibleLabels(k);
                   equivalence_labels{thisLabel} = unique([equivalence_labels{thisLabel} possibleLabels']);
                end
                                
            else
                % or assign it a label
                labels(i,j) = nextlabel;
                % and store the connected labels for future
                equivalence_labels{nextlabel} = nextlabel;
                nextlabel = nextlabel +1;
                
            end
        end
    end
end

% remove the background padding
labels = labels(2:end-1,2:end-1);
% initalize indexes that will be removed 
not_unique_idx = [];

% iterate through 
for i = 1:length(equivalence_labels)
    for j = 1:length(equivalence_labels)
        
        % combine the associated connected values 
        % and remove the repeated values
        if sum(ismember(equivalence_labels{i},equivalence_labels{j}))
            equivalence_labels{i} = union(equivalence_labels{i},equivalence_labels{j});
            if i~=j && j>i
                not_unique_idx = [not_unique_idx j];
            end
        end
    end
end

% keep only the labels that are connected
equivalence_labels(unique(not_unique_idx)) = [];
% initalize the final output
gray_labels = zeros(size(labels));

% iterate
for i = 1:length(equivalence_labels)
    
    % create each label as a new layer
    tempImg = zeros(size(labels));
    thisLinked = equivalence_labels{i};
    for j = 1:length(thisLinked)
        
        tempImg = tempImg + (labels == thisLinked(j));
        
    end
    
    % combine those layers as unique grayscale 
    gray_labels = gray_labels + double(tempImg)*i;
    
end

% will return gray_labels


end