function [dL,dW] = mj_nnwsum(L,W,dzdy, varargin)
% [dL,dW] = mj_nnwsum(L,W,dzdy, varargin)
% Weighted sum. Useful for multi-task learning
%
% Input:
%  - W: weights (one per task)
%  - L: loss vector (one per task)
%
% Options:
%  - 'learnW': boolean to learn weights (backprop)
%
% (c) MJMJ/2016

opts.learnW = false;
%opts.thr = 0.5;

opts = vl_argparse(opts,varargin) ;
%Y2 = []; % Init

%nloss = length(L); 

if isempty(dzdy) %forward
   Y = W * L';
   dL = Y; % Output
else   
   % Init
   dW = zerosLike(W);
   dL = zerosLike(L);
   
   % Derivative wrt Loss
   dL(:) = W .* dzdy;
   
   % Derivative wrt Weights
   if opts.learnW
      dW(:) = L .* dzdy;      
   end
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
    y = gpuArray.zeros(size(x),'single') ;
else
    y = zeros(size(x),'single') ;
end
