function [hdl, F] = mj_drawCNNfiltersMosaic(filters, drawpars)
% [hdl, F] = mj_drawCNNfiltersMosaic(filters)
% Draws filters learnt by a CNN compatible with MatConvNet
% COMMENT ME!!!
%
% Input:
%  - drawpars: struct
%      .maxfilters = inf; % All
%      .maxchannels = inf;
%      .color = false;
%      .independent = true;
%
%
% (c) MJMJ/2015

if ~exist('drawpars', 'var')
    drawpars = [];
end

if isempty(drawpars)
    drawpars.maxfilters = inf; % All
    drawpars.maxchannels = inf;
    drawpars.color = false;
    drawpars.independent = true;
    drawpars.singleChannel = false;
end

%% Create figure
hdl = figure;

nfilters = min( size(filters,4), drawpars.maxfilters);
%rc = floor(sqrt(nfilters));

%rc2 = rc*rc;

nch = min( size(filters,3), drawpars.maxchannels);

F = [];

for ix = 1:nfilters,
    %subplot(rc,rc,ix);
    %imagesc( filters(:,:,1,ix) );
    if drawpars.singleChannel
        fc1 = mj_compactFilter4Draw(filters(:,:,:,ix));
        
        if drawpars.independent
            mn = mean(fc1(:));
            ss = std(fc1(:));
            fc = (fc1-mn) / ss;
        else
            % Normalize values to show
            mn = mean(fc1(:));
            ss = std(fc1(:));
            fc = (fc1-mn) / ss;
        end
    elseif drawpars.color && nch == 75
        fc1 = mj_compactFilter4Draw(filters(:,:,1:3:nch,ix));
        fc2 = mj_compactFilter4Draw(filters(:,:,2:3:nch,ix));
        fc3 = mj_compactFilter4Draw(filters(:,:,3:3:nch,ix));
        
        if drawpars.independent
            mn = mean(fc1(:));
            ss = std(fc1(:));
            fc1 = (fc1-mn) / ss;
            
            mn = mean(fc2(:));
            ss = std(fc2(:));
            fc2 = (fc2-mn) / ss;
            
            mn = mean(fc3(:));
            ss = std(fc3(:));
            fc3 = (fc3-mn) / ss;
            
            fc = zeros(size(fc1, 1), size(fc1, 2), 3);
            fc(:,:,1) = fc1;
            fc(:,:,2) = fc2;
            fc(:,:,3) = fc3;
        else
            fc = [fc1, zeros(size(fc1,1),1), fc2, zeros(size(fc1,1),1), fc3];
            
            % Normalize values to show
            mn = mean(fc(:));
            ss = std(fc(:));
            fc = (fc-mn) / ss;
        end
    else
        fc1 = mj_compactFilter4Draw(filters(:,:,1:2:nch,ix));
        fc2 = mj_compactFilter4Draw(filters(:,:,2:2:nch,ix));
        
        if drawpars.independent
            mn = mean(fc1(:));
            ss = std(fc1(:));
            fc1 = (fc1-mn) / ss;
            
            mn = mean(fc2(:));
            ss = std(fc2(:));
            fc2 = (fc2-mn) / ss;
            
            fc = [fc1, zeros(size(fc1,1),1), fc2];
        else
            fc = [fc1, zeros(size(fc1,1),1), fc2];
            
            % Normalize values to show
            mn = mean(fc(:));
            ss = std(fc(:));
            fc = (fc-mn) / ss;
        end
    end
    
    if drawpars.color
        F = [F; fc; zeros(1, size(fc,2), 3)];
    else
        F = [F; fc];%; zeros(1, size(fc,2))];
    end
    %imagesc( fc );
    
end

% Display
imagesc(F);
axis image;
if ~drawpars.color
    colormap gray
end
axis off

