function scores = mj_classifyWithDAG(net, data, layertype, level, imdb, step)
% scores = mj_classifyWithDAG(net, data, layertype, level, imdb)
% Applies a trained net to data.
% Useful to get feature representations of input data.
%
% Input:
%  - net: trained CNN
%  - data: matrix [nrows, ncols, nchannels, nsamples]
%  - layertype: string with type of layer.
%
% Output:
%  - scores: matrix with output data
%
% See also vl_simplenn, mj_getCNNdataLayer
%
% (c) MJMJ/2016

if ~exist('level', 'var')
    level = 1;
end

if ~exist('step', 'var')
    step = 128;
end

% Check whether the model was computed at GPU
% if strcmp(class(net.layers{1}.weights{1}), 'gpuArray')
%     %if strcmp(class(net.layers{1}.filters), 'gpuArray')
% else
%     net = vl_simplenn_move(net, 'gpu');
% end
if gpuDeviceCount > 0
   net.move('gpu');
end

scores = [];
nused = 0;
if isempty(data)
    nsamples = imdb.images.sizes(4);
else
    nsamples = size(data,4);
end
inix = 1;
endix = min(step, nsamples);

while nused < nsamples
    if isempty(data)
        [im, ~, ~] = loadBatchH5(imdb, inix:endix);
        datachunk = im;
    else
        datachunk = data(:,:,:,inix:endix);
    end
    if strcmp(net.device, 'gpu')
       datachunk = gpuArray(datachunk);
    end
    
%     %restest = vl_simplenn(net, datachunk, [], [], 'disableDropout', true);
%     restest = vl_simplenn(net, datachunk);
    if ~strcmp(layertype, 'softmax') && ~strcmp(layertype, 'probs') 
       net.vars(net.getVarIndex(layertype)).precious = true;
       l = layertype;
    else
       l = layertype;  %'softmax'; was used by FC
    end
    net.eval({'input', datachunk});
    
    scores_ = net.vars(net.getVarIndex(l)).value ;
    scores_ = squeeze(gather(scores_)) ;    

    %scores_ = squeeze(gather(restest(end).x)) ;
%     if strcmp(layertype, 'regress')
%         l = 'dagnn.Conv';
%     else
%         l = layertype;
%     end
%     scores_ = mj_getDAGdataLayer(net, restest, l, level); scores_ = squeeze(scores_);
%     if strcmp(layertype, 'regress')
%         scores_ = scores_';
%     end
   if size(scores_,2) == 1
      scores_ = scores_';
   end
    scores = [scores, scores_];
    
    nused = nused + (endix-inix)+1;
    
    % Update positions
    inix = inix+step;
    endix = min(inix+step-1, nsamples);
end % while

% else
%    %restest = vl_simplenn(net, data, [], [], 'disableDropout', true);
%    restest = vl_simplenn(net, data);
%    %scores = squeeze(gather(restest(end).x));
%    scores = mj_getCNNdataLayer(net, restest, layertype, level);
%    scores = squeeze(scores);
% end
