function [ kittlerThreshold, H , thresholded_img] = HW4_thresh(varargin)
% Sam Freitas
% J. Kittler and J. Illingworth, "Minimum Error Thresholding,"
% Pattern Recognition, vol. 19, no. 1, pp. 41-47, 1986.
try
    % Check and see if the user inputted a threshold
    
    [img, user_threshold] = (varargin{:});
    
    disp(['User threshold detected, using value ' num2str(user_threshold)]);
    
    % Use and return the user threshold and images
    kittlerThreshold = user_threshold;
    
    % create thresholded image
    thresholded_img = (img>=user_threshold);
    
    % return H
    [thisHist,~] = imhist(img);
    thisHist = thisHist/length(img(:));
    H = thisHist;
    
catch
    % Use kittlers method if there isnt a threhsold given
    
    img = (varargin{:});
    
    % initalize H
    H = NaN * ones(256, 1);
    
    % Calc image histogram 
    [thisHist,~] = imhist(img);
    thisHist = thisHist/length(img(:));
    
    % Step through all the possible values 
    for T = 1:256
        
        % separate into two histograms assuming gaussian
        q1prev = thisHist(1:T);
        q2prev = thisHist((T+1):end);
        
        % find number of pixels in each hist section
        q1 = sum(q1prev);
        q2 = sum(q2prev);
        
        % if both histograms are populated continue
        if (q1 > 0) && (q2 > 0)
            
            % calculate the means
            mu1 = sum(q1prev .* (1:T)') / q1;
            mu2 = sum(q2prev .* (1:(256-T))') / q2;
            
            % compute the variances 
            var1 = (sum(q1prev .* (((1:T)' - mu1) .^2) ) / q1);
            var2 = (sum(q2prev .* (((1:(256-T))' - mu2) .^2) ) / q2);
            
            % if both variances are positive 
            if (var1 > 0) && (var2 > 0)
                
                % calculate the H value for that bin
                H(T-1) = (q1*log(var1) + q2*log(var2))/2 ...
                    - (q1*log(q1) + q2*log(q2));
                
            end
        end
        
    end
    
    % Find the min index of the histogram for optimal threshold
    [~, kittlerThreshold] = min(H);
    
    disp(['User threshold Not detected, using Kittlers method found optimal value ' num2str(kittlerThreshold)]);
    
    % create thresholded image
    thresholded_img = (img>=kittlerThreshold);
end
end