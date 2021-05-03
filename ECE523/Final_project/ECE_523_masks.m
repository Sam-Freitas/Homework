clear all; close all;
files = dir("*.png");
for i = 1:numel(files);
    filename = files(i).name;
    mri = imread(filename);
    mri_bw = im2bw(mri, 0.25);
    Label = bwlabel(mri_bw); 
    s = regionprops(Label,'Area','BoundingBox');
    area_values = [s.Area];
    idx = find((5000 <= area_values) & (area_values <= 30000));
    brain = ismember(Label, idx);
    brain = double(brain);
    brain = imfill(brain, 'holes');
    mask_name = regexprep(filename,'.png','_mask.png');
    imwrite(brain, mask_name, 'BitDepth',16);
end